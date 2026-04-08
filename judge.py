"""
judge.py — LLM-as-a-Judge: Generic, Prompt-Registry-Driven Evaluation Engine
─────────────────────────────────────────────────────────────────────────────
Evaluates the quality of SCRIBE complaint summaries.

All prompt text lives in config/prompts.yaml — this file contains ZERO prompt
strings. It only contains the mechanics of calling LLMs, parsing responses,
computing weighted scores, and applying strict verdicts.

Flow:
  Judge(profile, prompt_loader)
    └── grade(case)
           ├── _resolve_criteria(case)       # apply case-level overrides
           ├── _group_by_prompt(criteria)    # single vs combined
           ├── LLM calls via _get_chain()    # lazy-cached, on-demand model init
           ├── weighted average
           └── _verdict()                   # Option B — strict

Adding a new evaluation criterion requires:
  1. A new prompt entry in config/prompts.yaml
  2. A new criterion block in config/evaluation_profiles.yaml
  3. Zero Python changes here.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from profile_loader import Profile, Criterion, PromptLoader


# ── Non-LLM deterministic metrics ─────────────────────────────────────────────

def compression_ratio(complaint: str, summary: str) -> float:
    """Length of summary / length of complaint. Ideal range: 0.10–0.35."""
    c = len(complaint.split())
    s = len(summary.split())
    return round(s / c, 2) if c > 0 else 0.0


def entity_recall(complaint: str, summary: str) -> float:
    """
    Extracts key entities (amounts, dates, IDs) from the complaint
    and checks how many appear in the summary.
    """
    pattern = (
        r'\$[\d,\.]+|'
        r'\b\d+[\s-]?(days?|weeks?|months?|years?|times?)\b|'
        r'\b(january|february|march|april|may|june|july|august|'
        r'september|october|november|december)\s+\d+\w*\b'
    )
    entities = re.findall(pattern, complaint.lower())
    entities = [e if isinstance(e, str) else ' '.join(e).strip() for e in entities]
    entities = [e for e in entities if e.strip()]

    if not entities:
        return 1.0
    found = sum(1 for e in entities if e.lower() in summary.lower())
    return round(found / len(entities), 2)


def compression_flag(ratio: float) -> str:
    if ratio < 0.10: return "too_short"
    if ratio > 0.40: return "too_long"
    return "ok"


# ── Judge ─────────────────────────────────────────────────────────────────────

class Judge:
    """
    Generic, prompt-registry-driven LLM judge for SCRIBE summary evaluation.

    All prompt text comes from PromptLoader (config/prompts.yaml).
    This class has no knowledge of specific criteria — it processes whatever
    criteria a Profile defines, using whatever prompt each criterion references.

    Per-case override support:
        Any case dict can include an "overrides" key to patch individual
        criterion settings for that one case only. The profile is never mutated.

        "overrides": {
            "entity_preservation": {"enabled": False},
            "faithfulness":        {"judge_model": "gemma3:4b", "weight": 2.5},
            "coverage":            {"prompt": "coverage_strict"},
        }
    """

    def __init__(self, profile: Profile, prompt_loader: Optional[PromptLoader] = None):
        self.profile       = profile
        self.prompt_loader = prompt_loader or PromptLoader()
        self._parser       = StrOutputParser()

        # LLM cache: model_name → ChatOllama instance
        # Pre-warm for all models declared in the profile;
        # extended on demand if a case override introduces a new model.
        self._llms: dict[str, ChatOllama] = {
            model: ChatOllama(model=model, temperature=0)
            for model in profile.unique_models
        }

        # Chain cache: (prompt_template_id, model_name) → runnable chain
        self._chain_cache: dict[tuple, object] = {}

    # ── Chain factory (lazy, cached) ───────────────────────────────────────────

    def _get_chain(self, prompt_template, model_name: str):
        """
        Return a cached (prompt | llm | parser) chain.
        Creates a new ChatOllama instance on demand for override-introduced models.
        """
        if model_name not in self._llms:
            self._llms[model_name] = ChatOllama(model=model_name, temperature=0)
        key = (id(prompt_template), model_name)
        if key not in self._chain_cache:
            self._chain_cache[key] = prompt_template | self._llms[model_name] | self._parser
        return self._chain_cache[key]

    # ── Criteria resolver (applies case-level overrides) ──────────────────────

    def _resolve_criteria(self, case: dict) -> dict:
        """
        Merge profile criteria with case["overrides"].
        The profile is never mutated — overrides apply per-call only.
        """
        overrides = case.get("overrides", {})
        if not overrides:
            return self.profile.criteria  # fast path

        effective = {}
        for crit_name, crit in self.profile.criteria.items():
            ov = overrides.get(crit_name, {})
            if ov:
                effective[crit_name] = Criterion(
                    name           = crit.name,
                    enabled        = ov.get("enabled",        crit.enabled),
                    judge_model    = ov.get("judge_model",    crit.judge_model),
                    prompt         = ov.get("prompt",         crit.prompt),
                    weight         = float(ov.get("weight",   crit.weight)),
                    pass_threshold = ov.get("pass_threshold", crit.pass_threshold),
                    fail_threshold = ov.get("fail_threshold", crit.fail_threshold),
                )
            else:
                effective[crit_name] = crit
        return effective

    # ── Prompt grouping ───────────────────────────────────────────────────────

    def _group_by_prompt(self, criteria: dict) -> tuple[list, dict]:
        """
        Split enabled criteria into:
          singles  : list of (crit_name, criterion) — one LLM call each
          combined : dict of prompt_key → list of crit_names — one shared LLM call

        Combined prompts (type=combined in prompts.yaml) are called once per
        unique prompt key, and their output is distributed across all criteria
        that reference that prompt.
        """
        singles:  list = []
        combined: dict = {}  # prompt_key → [crit_name, ...]

        for crit_name, crit in criteria.items():
            if not crit.enabled:
                continue
            if self.prompt_loader.is_combined(crit.prompt_key):
                combined.setdefault(crit.prompt_key, []).append(crit_name)
            else:
                singles.append((crit_name, crit))

        return singles, combined

    # ── Parsing helpers ────────────────────────────────────────────────────────

    def _parse(self, text: str) -> dict:
        try:
            m = re.search(r'\{.*\}', text, re.DOTALL)
            if m:
                clean = re.sub(r'[\x00-\x1f\x7f]', ' ', m.group())
                return json.loads(clean)
        except Exception:
            pass
        return {}

    def _norm(self, score, out_of: int = 5) -> float:
        """Normalise a 1–5 scale score to 0.0–1.0."""
        try:
            return round(float(score) / out_of, 2)
        except (TypeError, ValueError):
            return 0.0

    # ── Main grading method ────────────────────────────────────────────────────

    def grade(self, case: dict, matched_rule: str = "default") -> dict:
        """
        Grade a single complaint case.

        Args:
            case         : dict with complaint, scribe_summary, and optional "overrides"
            matched_rule : policy name that selected this profile (for audit trail)

        Returns:
            Full result dict: scores, reasons, weights, models_used, verdict, raw outputs.
        """
        complaint = case["complaint"]
        summary   = case["scribe_summary"]
        criteria  = self._resolve_criteria(case)
        overridden = [k for k in case.get("overrides", {}) if k in criteria]

        scores:      dict[str, float] = {}
        reasons:     dict[str, str]   = {}
        weights:     dict[str, float] = {}
        models_used: dict[str, str]   = {}
        raw:         dict[str, str]   = {}

        inputs = {"complaint": complaint, "summary": summary}

        # ── Split criteria into single-call and combined-call groups ──────────
        singles, combined_groups = self._group_by_prompt(criteria)

        # ── Process single-output criteria ────────────────────────────────────
        for crit_name, crit in singles:
            prompt = self.prompt_loader.get(crit.prompt_key)
            chain  = self._get_chain(prompt.template, crit.judge_model)
            raw_out = chain.invoke(inputs)
            parsed  = self._parse(raw_out)

            scores[crit_name]      = self._norm(parsed.get("score", 1))
            reasons[crit_name]     = parsed.get("reasoning", "")
            weights[crit_name]     = crit.weight
            models_used[crit_name] = crit.judge_model
            raw[crit_name]         = raw_out

        # ── Process combined-output criteria ──────────────────────────────────
        # One LLM call per unique combined prompt; then distribute sub-scores.
        for prompt_key, crit_names in combined_groups.items():
            prompt     = self.prompt_loader.get(prompt_key)
            # Use the model from the first criterion in this group
            first_crit = criteria[crit_names[0]]
            chain      = self._get_chain(prompt.template, first_crit.judge_model)
            raw_out    = chain.invoke(inputs)
            parsed     = self._parse(raw_out)
            raw[prompt_key] = raw_out

            for crit_name in crit_names:
                crit = criteria[crit_name]
                sub  = parsed.get(crit_name, {})
                scores[crit_name]      = self._norm(sub.get("score", 1))
                reasons[crit_name]     = sub.get("reason", "")
                weights[crit_name]     = crit.weight
                models_used[crit_name] = crit.judge_model

        # ── Non-LLM deterministic metrics ────────────────────────────────────
        ratio  = compression_ratio(complaint, summary)
        recall = entity_recall(complaint, summary)
        flag   = compression_flag(ratio)

        # ── Weighted average ──────────────────────────────────────────────────
        if scores:
            total_weight = sum(weights[k] for k in scores)
            weighted_avg = round(
                sum(scores[k] * weights[k] for k in scores) / total_weight, 2
            )
        else:
            weighted_avg = 0.0

        # ── Verdict (Option B — strict) ───────────────────────────────────────
        verdict = self._verdict(scores, weighted_avg, criteria)

        return {
            "id":                 case["id"],
            "profile_name":       self.profile.name,
            "matched_rule":       matched_rule,
            "scores":             scores,
            "reasons":            reasons,
            "weights":            weights,
            "models_used":        models_used,
            "weighted_avg":       weighted_avg,
            "non_llm": {
                "compression_ratio": ratio,
                "compression_flag":  flag,
                "entity_recall":     recall,
            },
            "verdict":             verdict,
            "raw":                 raw,
            "disabled_criteria":   [k for k, c in criteria.items() if not c.enabled],
            "overridden_criteria": overridden,
        }

    # ── Verdict logic (Option B — strict) ─────────────────────────────────────

    def _verdict(
        self,
        scores:       dict[str, float],
        weighted_avg: float,
        criteria:     dict,
    ) -> str:
        """
        PASS         → weighted_avg >= pass_threshold
                       AND no criterion below its effective fail_threshold
        FAIL         → weighted_avg < fail_threshold
                       OR any criterion below its effective fail_threshold
        NEEDS_REVIEW → everything else
        """
        p = self.profile.pass_threshold
        f = self.profile.fail_threshold

        any_criterion_failed = any(
            score < criteria[name].effective_fail(f)
            for name, score in scores.items()
            if name in criteria
        )

        if weighted_avg < f or any_criterion_failed:
            return "FAIL"
        if weighted_avg >= p and not any_criterion_failed:
            return "PASS"
        return "NEEDS_REVIEW"
