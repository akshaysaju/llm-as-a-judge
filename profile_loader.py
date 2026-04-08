"""
profile_loader.py — YAML Config Loader for SCRIBE Evaluation
─────────────────────────────────────────────────────────────
Loads two config files:
  1. config/evaluation_profiles.yaml  — defines HOW to evaluate (criteria, models, thresholds)
  2. config/policies.yaml             — defines WHICH profile to apply based on business conditions

Resolves a complaint case to its matching evaluation profile via:
  resolve_profile(case) → (Profile, policy_name)

Policy matching (first-match wins, top → bottom):
  severity : matches case["severity"] field (case-insensitive)
  keywords : OR-matched against the full complaint text (case-insensitive substring)
  Multiple conditions in one policy are AND-ed.
  No match → default_profile (typically "standard")
"""

from __future__ import annotations

import os
import re
import yaml
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate



# ── Config file paths ──────────────────────────────────────────────────────────

_CONFIG_DIR    = os.path.join(os.path.dirname(__file__), "config")
_PROFILES_FILE = os.path.join(_CONFIG_DIR, "evaluation_profiles.yaml")
_POLICIES_FILE = os.path.join(_CONFIG_DIR, "policies.yaml")
_PROMPTS_FILE  = os.path.join(_CONFIG_DIR, "prompts.yaml")



# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Criterion:
    name:           str
    enabled:        bool
    judge_model:    str
    prompt:         str           = ""     # empty = use criterion name as key in prompts.yaml
    weight:         float         = 1.0
    pass_threshold: Optional[float] = None
    fail_threshold: Optional[float] = None

    @property
    def prompt_key(self) -> str:
        """Prompt name to look up in prompts.yaml. Defaults to criterion name."""
        return self.prompt or self.name

    def effective_pass(self, profile_pass: float) -> float:
        return self.pass_threshold if self.pass_threshold is not None else profile_pass

    def effective_fail(self, profile_fail: float) -> float:
        return self.fail_threshold if self.fail_threshold is not None else profile_fail



@dataclass
class Profile:
    name:           str
    description:    str
    pass_threshold: float
    fail_threshold: float
    criteria:       dict[str, Criterion] = field(default_factory=dict)

    @property
    def unique_models(self) -> list[str]:
        """All distinct Ollama model names referenced by enabled criteria."""
        return sorted({
            c.judge_model
            for c in self.criteria.values()
            if c.enabled and c.judge_model
        })


@dataclass
class Prompt:
    name:        str
    type:        str              # "single" or "combined"
    description: str
    template:    ChatPromptTemplate
    outputs:     list[str] = field(default_factory=list)  # only used for type=combined


@dataclass
class Policy:
    name:         str
    description:  str
    eval_profile: str
    severity:     Optional[str]   = None   # low / medium / high
    keywords:     list[str]       = field(default_factory=list)



# ── PromptLoader ──────────────────────────────────────────────────────────────

class PromptLoader:
    """
    Loads config/prompts.yaml and provides prompt templates by name.

    Prompt types:
        single   — one LLM call, returns {"reasoning": "...", "score": 1-5}
        combined — one LLM call, returns multiple scores; outputs declares which criteria

    Template variables ({complaint} and {summary}) are substituted at invocation time.
    Literal { and } in the template text (e.g. JSON examples) are auto-escaped so that
    ChatPromptTemplate does not treat them as variables.

    Usage:
        prompt_loader = PromptLoader()
        prompt = prompt_loader.get("faithfulness")      # Prompt dataclass
        prompt = prompt_loader.get("quality_combined")  # combined type
    """

    _KNOWN_VARS = re.compile(r'\{(complaint|summary)\}')

    def __init__(self, prompts_path: str = _PROMPTS_FILE):
        with open(prompts_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        self._prompts: dict[str, Prompt] = {}
        for name, data in raw.get("prompts", {}).items():
            template_str = data.get("template", "")
            self._prompts[name] = Prompt(
                name        = name,
                type        = data.get("type", "single"),
                description = data.get("description", ""),
                template    = ChatPromptTemplate.from_template(
                    self._escape(template_str)
                ),
                outputs     = data.get("outputs", []),
            )

    def _escape(self, text: str) -> str:
        """
        Escape all literal { and } in the template so ChatPromptTemplate treats
        them as plain text, then restore only the known variable placeholders.
        Result: {complaint} and {summary} remain as substitution targets;
        everything else (e.g. JSON examples) is escaped to {{ and }}.
        """
        # Step 1: escape all braces
        escaped = text.replace("{", "{{").replace("}", "}}")
        # Step 2: restore known template variables
        escaped = escaped.replace("{{complaint}}", "{complaint}")
        escaped = escaped.replace("{{summary}}",   "{summary}")
        return escaped

    def get(self, name: str) -> Prompt:
        """Return the Prompt for the given name. Raises KeyError if not found."""
        if name not in self._prompts:
            raise KeyError(
                f"Prompt '{name}' not found in prompts.yaml. "
                f"Available: {list(self._prompts.keys())}"
            )
        return self._prompts[name]

    def is_combined(self, name: str) -> bool:
        """True if this prompt returns multiple scores in one call."""
        return self._prompts.get(name, Prompt("", "single", "", None)).type == "combined"

    @property
    def names(self) -> list[str]:
        return list(self._prompts.keys())


# ── ProfileLoader ─────────────────────────────────────────────────────────────

class ProfileLoader:
    """
    Loads all three config files and provides resolved profiles per complaint.

    Usage:
        loader = ProfileLoader()
        profile, policy_name = loader.resolve_profile(case)
        judge = Judge(profile, loader.prompt_loader)
    """

    def __init__(
        self,
        profiles_path: str = _PROFILES_FILE,
        policies_path:  str = _POLICIES_FILE,
        prompts_path:   str = _PROMPTS_FILE,
    ):
        with open(profiles_path, encoding="utf-8") as f:
            raw_profiles = yaml.safe_load(f)
        with open(policies_path, encoding="utf-8") as f:
            raw_policies = yaml.safe_load(f)

        self.prompt_loader:   PromptLoader       = PromptLoader(prompts_path)
        self.profiles:        dict[str, Profile]  = self._parse_profiles(raw_profiles)
        self.policies:        list[Policy]         = self._parse_policies(raw_policies)
        self.default_profile: str                  = raw_policies.get("default_profile", "standard")

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse_profiles(self, raw: dict) -> dict[str, Profile]:
        profiles = {}
        for name, data in raw.get("profiles", {}).items():
            criteria = {}
            for crit_name, crit_data in data.get("criteria", {}).items():
                criteria[crit_name] = Criterion(
                    name           = crit_name,
                    enabled        = bool(crit_data.get("enabled", True)),
                    judge_model    = crit_data.get("judge_model", "mistral"),
                    prompt         = crit_data.get("prompt", ""),  # empty → defaults to crit name
                    weight         = float(crit_data.get("weight", 1.0)),
                    pass_threshold = crit_data.get("pass_threshold"),
                    fail_threshold = crit_data.get("fail_threshold"),
                )
            profiles[name] = Profile(
                name           = name,
                description    = data.get("description", ""),
                pass_threshold = float(data.get("pass_threshold", 0.75)),
                fail_threshold = float(data.get("fail_threshold", 0.50)),
                criteria       = criteria,
            )
        return profiles

    def _parse_policies(self, raw: dict) -> list[Policy]:
        policies = []
        for p in raw.get("policies", []):
            conditions = p.get("conditions", {})
            policies.append(Policy(
                name         = p.get("name", "unnamed"),
                description  = p.get("description", ""),
                eval_profile = p.get("eval_profile", "standard"),
                severity     = conditions.get("severity"),
                keywords     = [kw.lower() for kw in conditions.get("keywords", [])],
            ))
        return policies

    # ── Profile resolution ────────────────────────────────────────────────────

    def resolve_profile(self, case: dict) -> tuple[Profile, str]:
        """
        Given a complaint case dict, find the first matching policy and return
        its evaluation Profile and the policy name (for audit/reporting).

        Matching logic (per policy, top → bottom, first match wins):
          severity match : case["severity"].lower() == policy.severity.lower()
          keywords match : ANY keyword is a substring of complaint text (case-insensitive)
          Multiple conditions : AND-ed (all must match)

        Falls back to default_profile if no policy matches.
        """
        complaint_text = (case.get("complaint", "") + " " + case.get("scribe_summary", "")).lower()
        case_severity  = (case.get("severity", "") or "").lower().strip()

        for policy in self.policies:
            severity_match = True
            keyword_match  = True

            # Severity condition (if specified)
            if policy.severity:
                severity_match = (case_severity == policy.severity.lower())

            # Keywords condition — ANY keyword found in complaint text (OR logic)
            if policy.keywords:
                keyword_match = any(kw in complaint_text for kw in policy.keywords)

            if severity_match and keyword_match:
                profile_name = policy.eval_profile
                if profile_name in self.profiles:
                    return self.profiles[profile_name], policy.name
                # Unknown profile name in policy — fall through to default
                break

        # No match — use default
        default = self.profiles.get(self.default_profile)
        if default is None:
            raise ValueError(
                f"default_profile '{self.default_profile}' not found in evaluation_profiles.yaml"
            )
        return default, "default"
