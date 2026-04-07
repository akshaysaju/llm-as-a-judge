"""
judge.py — LLM as a Judge (Decoupled, Profile-Driven, Override-Aware)
──────────────────────────────────────────────────────────────────────
Grades SCRIBE (summary) and TRACE (classification) outputs against a
dynamically loaded evaluation Profile.

Key design points:
  • Judge takes a Profile — all thresholds, criteria, and models from YAML.
  • Chains are built LAZILY per (prompt, model) pair and cached.
    → Any model introduced by a case-level override is created on demand.
  • _resolve_criteria(case) merges profile criteria with case["overrides"].
    → The profile is NEVER mutated; overrides are applied per grade() call.
  • Per-case override fields (all optional, any combination):
        enabled          — True/False (turn metric on or off for this case)
        judge_model      — switch to a different Ollama model for this criterion
        weight           — change contribution to weighted average
        pass_threshold   — tighten/loosen per-criterion pass gate
        fail_threshold   — tighten/loosen per-criterion fail gate

  Example in cases.py:
      "overrides": {
          "entity_preservation": {"enabled": False},
          "faithfulness":        {"judge_model": "gemma3:4b", "weight": 2.5},
      }

  Verdict logic (Option B — strict):
      PASS         → weighted_avg >= profile.pass_threshold
                     AND no criterion score < its effective fail_threshold
      FAIL         → weighted_avg < profile.fail_threshold
                     OR any criterion score < its effective fail_threshold
      NEEDS_REVIEW → everything else
"""

from __future__ import annotations

import json
import re
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from profile_loader import Profile, Criterion


# ── Valid TRACE categories ─────────────────────────────────────────────────────

VALID_CATEGORIES = [
    "Billing/Fees Dispute",
    "Fraud/Unauthorised Transaction",
    "Fraud/Identity Theft",
    "Loan/Mortgage Complaint",
    "Account Access Issue",
    "Customer Service Complaint",
    "Other",
]


# ── Prompts ───────────────────────────────────────────────────────────────────

FAITHFULNESS_PROMPT = ChatPromptTemplate.from_template("""
You are evaluating a complaint summary for faithfulness.

Step 1 — List every factual claim in the summary.
Step 2 — For each claim, check if it is present in the original complaint.
Step 3 — Score using this rubric:
  1 = Summary contains facts not present in the complaint (hallucination)
  2 = Summary has significant paraphrasing that distorts meaning
  3 = Summary is mostly faithful, minor wording differences only
  4 = All claims traceable but one small detail is off
  5 = Every claim in the summary can be traced directly to the complaint

Original Complaint:
{complaint}

SCRIBE Summary:
{summary}

Reply with JSON only. Example: {{"reasoning": "your step-by-step analysis", "score": 4}}
Use an actual integer for score.
""")

COVERAGE_PROMPT = ChatPromptTemplate.from_template("""
You are evaluating a complaint summary for coverage.

Step 1 — Identify in the complaint: (a) core problem, (b) impact on customer, (c) urgency signals, (d) what the customer is asking for.
Step 2 — Check which of these appear in the summary.
Step 3 — Score using this rubric:
  1 = Core problem is missing from the summary
  2 = Core problem present but customer ask or impact is missing
  3 = Core problem and ask present but urgency or impact is missing
  4 = All key parts present, one minor detail missing
  5 = Summary fully captures problem, impact, urgency signals, and customer ask

Original Complaint:
{complaint}

SCRIBE Summary:
{summary}

Reply with JSON only. Example: {{"reasoning": "your step-by-step analysis", "score": 3}}
Use an actual integer for score.
""")

ACTION_PROMPT = ChatPromptTemplate.from_template("""
You are evaluating whether a complaint summary is action-oriented.

Step 1 — Identify what specific action the customer wants (refund, fix, callback, investigation, etc.).
Step 2 — Check if a support agent reading only the summary would immediately know what to do next.
Step 3 — Score using this rubric:
  1 = Summary gives no indication of what action is needed
  2 = Action is vaguely implied but not clear
  3 = Action is mentioned but lacks specifics (amount, date, account)
  4 = Action is clear with most relevant details
  5 = Agent knows exactly what to do from the summary alone — action, amount, date, account all present

Original Complaint:
{complaint}

SCRIBE Summary:
{summary}

Reply with JSON only. Example: {{"reasoning": "your step-by-step analysis", "score": 5}}
Use an actual integer for score.
""")

COMBINED_PROMPT = ChatPromptTemplate.from_template("""
You are evaluating a complaint summary on four dimensions. Score each 1–5.

Rubrics:
  Conciseness         — 1: too long or too vague  3: acceptable length  5: perfectly compressed
  Clarity             — 1: confusing or ambiguous  3: mostly clear  5: agent understands in 5 seconds
  Urgency Tone        — 1: angry complaint summarised as calm  3: partial  5: tone and severity match original
  Entity Preservation — 1: amounts/dates/IDs missing  3: some preserved  5: all key entities retained

Original Complaint:
{complaint}

SCRIBE Summary:
{summary}

Reply with JSON only:
{{
  "conciseness":          {{"score": 1-5, "reason": "one sentence"}},
  "clarity":              {{"score": 1-5, "reason": "one sentence"}},
  "urgency_tone":         {{"score": 1-5, "reason": "one sentence"}},
  "entity_preservation":  {{"score": 1-5, "reason": "one sentence"}}
}}
""")

URGENCY_PROMPT = ChatPromptTemplate.from_template("""
You are evaluating whether a complaint summary preserves urgency tone correctly.

Step 1 — Assess the tone and severity of the original complaint (frustrated? urgent? calm?).
Step 2 — Check if the summary reflects the same emotional tone and severity.
Step 3 — Score using this rubric:
  1 = Urgent/angry complaint is summarised as calm or neutral
  2 = Severity is partially reflected but significantly softened
  3 = Tone is partially preserved but could be stronger
  4 = Tone and severity mostly match with minor softening
  5 = Summary tone exactly matches the urgency level of the original complaint

Original Complaint:
{complaint}

SCRIBE Summary:
{summary}

Reply with JSON only. Example: {{"reasoning": "your analysis", "score": 4}}
Use an actual integer for score.
""")

TRACE_PROMPT = ChatPromptTemplate.from_template("""
You are grading a complaint classification produced by TRACE.

Step 1 — Identify the primary nature of the complaint.
Step 2 — Check if the assigned category matches.
Step 3 — Assign a score:
  Use 1.0 if the category is correct.
  Use 0.5 if it is a reasonable alternative but not the best fit.
  Use 0.0 if it is clearly wrong.

Original Complaint:
{complaint}

TRACE assigned: {category}
Valid categories: {valid_categories}

Reply with JSON only. Example format: {{"reasoning": "one sentence", "score": 1.0}}
Use an actual number for score, not a range.
""")


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
    Profile-driven LLM Judge with per-case override support.

    Args:
        profile: A Profile dataclass loaded by ProfileLoader.
                 Provides the base configuration (criteria, models, thresholds).

    Per-case override example (in cases.py):
        "overrides": {
            "entity_preservation": {"enabled": False},
            "faithfulness":        {"judge_model": "gemma3:4b", "weight": 2.5},
            "coverage":            {"pass_threshold": 0.85, "fail_threshold": 0.65},
        }

    Overridable fields (per criterion, all optional):
        enabled          — True/False to turn a metric on or off for this case
        judge_model      — any Ollama model name; new instance created on demand
        weight           — contribution to the weighted average
        pass_threshold   — per-criterion pass cutoff override
        fail_threshold   — per-criterion fail cutoff override (hard FAIL if breached)
    """

    def __init__(self, profile: Profile):
        self.profile = profile
        self._parser = StrOutputParser()

        # LLM instances keyed by model name — pre-warmed for profile models,
        # and extended on demand when case-level overrides introduce new models.
        self._llms: dict[str, ChatOllama] = {
            model_name: ChatOllama(model=model_name, temperature=0)
            for model_name in profile.unique_models
        }

        # Chain cache: (id(prompt_template), model_name) → runnable chain.
        # Avoids rebuilding the same prompt|llm|parser triple repeatedly.
        self._chain_cache: dict[tuple, object] = {}

    # ── Chain factory (lazy, cached) ───────────────────────────────────────────

    def _get_chain(self, prompt, model_name: str):
        """
        Return a cached (prompt | llm | parser) chain for the given model.
        Creates a new ChatOllama instance if the model hasn't been seen before
        (e.g. introduced by a case-level override pointing to a new model).
        """
        if model_name not in self._llms:
            self._llms[model_name] = ChatOllama(model=model_name, temperature=0)
        key = (id(prompt), model_name)
        if key not in self._chain_cache:
            self._chain_cache[key] = prompt | self._llms[model_name] | self._parser
        return self._chain_cache[key]

    # ── Criteria resolver ─────────────────────────────────────────────────────

    def _resolve_criteria(self, case: dict) -> dict:
        """
        Merge profile-level criteria with case["overrides"].

        Returns a fresh dict of Criterion objects — the profile is NEVER mutated.
        Any field not present in the override inherits the profile's value.
        """
        overrides = case.get("overrides", {})
        if not overrides:
            return self.profile.criteria   # fast path — nothing to merge

        effective = {}
        for crit_name, crit in self.profile.criteria.items():
            ov = overrides.get(crit_name, {})
            if ov:
                effective[crit_name] = Criterion(
                    name           = crit.name,
                    enabled        = ov.get("enabled",        crit.enabled),
                    judge_model    = ov.get("judge_model",    crit.judge_model),
                    weight         = float(ov.get("weight",   crit.weight)),
                    pass_threshold = ov.get("pass_threshold", crit.pass_threshold),
                    fail_threshold = ov.get("fail_threshold", crit.fail_threshold),
                )
            else:
                effective[crit_name] = crit
        return effective

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
        Grade a single complaint case using the configured profile +
        any case-level overrides in case["overrides"].

        Args:
            case         : dict with complaint, scribe_summary, trace_category,
                           and optionally "overrides" for per-case patches.
            matched_rule : which routing rule selected this profile (for traceability).

        Returns:
            Full result dict: scores, reasons, weights, models_used, overridden_criteria,
            verdict, non_llm metrics, raw LLM responses.
        """
        complaint = case["complaint"]
        summary   = case["scribe_summary"]

        # Effective criteria = profile base + case overrides (profile never mutated)
        criteria = self._resolve_criteria(case)

        scores:      dict[str, float] = {}
        reasons:     dict[str, str]   = {}
        weights:     dict[str, float] = {}
        models_used: dict[str, str]   = {}
        raw:         dict[str, str]   = {}

        # Track which criteria were patched by case-level overrides (for reporting)
        overridden = [k for k in case.get("overrides", {}) if k in criteria]

        # ── Faithfulness ──────────────────────────────────────────────────────
        crit = criteria.get("faithfulness")
        if crit and crit.enabled:
            chain = self._get_chain(FAITHFULNESS_PROMPT, crit.judge_model)
            raw_f = chain.invoke({"complaint": complaint, "summary": summary})
            f     = self._parse(raw_f)
            scores["faithfulness"]      = self._norm(f.get("score", 1))
            reasons["faithfulness"]     = f.get("reasoning", "")
            weights["faithfulness"]     = crit.weight
            models_used["faithfulness"] = crit.judge_model
            raw["faithfulness"]         = raw_f

        # ── Coverage ──────────────────────────────────────────────────────────
        crit = criteria.get("coverage")
        if crit and crit.enabled:
            chain = self._get_chain(COVERAGE_PROMPT, crit.judge_model)
            raw_c = chain.invoke({"complaint": complaint, "summary": summary})
            c     = self._parse(raw_c)
            scores["coverage"]      = self._norm(c.get("score", 1))
            reasons["coverage"]     = c.get("reasoning", "")
            weights["coverage"]     = crit.weight
            models_used["coverage"] = crit.judge_model
            raw["coverage"]         = raw_c

        # ── Action-Orientedness ───────────────────────────────────────────────
        crit = criteria.get("action_orientedness")
        if crit and crit.enabled:
            chain = self._get_chain(ACTION_PROMPT, crit.judge_model)
            raw_a = chain.invoke({"complaint": complaint, "summary": summary})
            a     = self._parse(raw_a)
            scores["action_orientedness"]      = self._norm(a.get("score", 1))
            reasons["action_orientedness"]     = a.get("reasoning", "")
            weights["action_orientedness"]     = crit.weight
            models_used["action_orientedness"] = crit.judge_model
            raw["action_orientedness"]         = raw_a

        # ── Combined block: conciseness, clarity, entity_preservation ─────────
        # Single LLM call handles all three. Model = first enabled criterion's model.
        combined_group = ["conciseness", "clarity", "entity_preservation"]
        enabled_combined = [
            n for n in combined_group
            if criteria.get(n) and criteria[n].enabled
        ]
        raw_cmb = {}
        if enabled_combined:
            combined_model = criteria[enabled_combined[0]].judge_model
            chain        = self._get_chain(COMBINED_PROMPT, combined_model)
            raw_combined = chain.invoke({"complaint": complaint, "summary": summary})
            raw_cmb      = self._parse(raw_combined)
            raw["combined"] = raw_combined

        for key in combined_group:
            crit = criteria.get(key)
            if crit and crit.enabled:
                scores[key]      = self._norm(raw_cmb.get(key, {}).get("score", 1))
                reasons[key]     = raw_cmb.get(key, {}).get("reason", "")
                weights[key]     = crit.weight
                models_used[key] = crit.judge_model

        # ── Urgency Tone (standalone prompt) ──────────────────────────────────
        crit = criteria.get("urgency_tone")
        if crit and crit.enabled:
            chain = self._get_chain(URGENCY_PROMPT, crit.judge_model)
            raw_u = chain.invoke({"complaint": complaint, "summary": summary})
            u     = self._parse(raw_u)
            scores["urgency_tone"]      = self._norm(u.get("score", 1))
            reasons["urgency_tone"]     = u.get("reasoning", "")
            weights["urgency_tone"]     = crit.weight
            models_used["urgency_tone"] = crit.judge_model
            raw["urgency_tone"]         = raw_u

        # ── TRACE classification ───────────────────────────────────────────────
        crit = criteria.get("trace_classification")
        if crit and crit.enabled:
            chain = self._get_chain(TRACE_PROMPT, crit.judge_model)
            raw_t = chain.invoke({
                "complaint":        complaint,
                "category":         case.get("trace_category", ""),
                "valid_categories": ", ".join(VALID_CATEGORIES),
            })
            t = self._parse(raw_t)
            # TRACE returns 0.0 / 0.5 / 1.0 directly — not 1–5 scale
            scores["trace_classification"]      = round(float(t.get("score", 0)), 2)
            reasons["trace_classification"]     = t.get("reasoning", "")
            weights["trace_classification"]     = crit.weight
            models_used["trace_classification"] = crit.judge_model
            raw["trace_classification"]         = raw_t

        # ── Non-LLM deterministic metrics ──────────────────────────────────────
        ratio  = compression_ratio(complaint, summary)
        recall = entity_recall(complaint, summary)
        flag   = compression_flag(ratio)

        # ── Weighted average score ─────────────────────────────────────────────
        if scores:
            total_weight = sum(weights[k] for k in scores)
            weighted_avg = round(
                sum(scores[k] * weights[k] for k in scores) / total_weight, 2
            )
        else:
            weighted_avg = 0.0

        # ── Verdict (Option B — strict) ────────────────────────────────────────
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
        scores:      dict[str, float],
        weighted_avg: float,
        criteria:    dict,
    ) -> str:
        """
        PASS:         weighted_avg >= profile.pass_threshold
                      AND no criterion below its effective fail_threshold
        FAIL:         weighted_avg < profile.fail_threshold
                      OR any criterion below its effective fail_threshold
        NEEDS_REVIEW: everything else
        """
        p = self.profile.pass_threshold
        f = self.profile.fail_threshold

        any_criterion_failed = any(
            score < criteria[crit_name].effective_fail(f)
            for crit_name, score in scores.items()
            if crit_name in criteria
        )

        if weighted_avg < f or any_criterion_failed:
            return "FAIL"
        if weighted_avg >= p and not any_criterion_failed:
            return "PASS"
        return "NEEDS_REVIEW"
