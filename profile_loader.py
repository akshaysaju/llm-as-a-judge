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
import yaml
from dataclasses import dataclass, field
from typing import Optional


# ── Config file paths ──────────────────────────────────────────────────────────

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")
_PROFILES_FILE = os.path.join(_CONFIG_DIR, "evaluation_profiles.yaml")
_POLICIES_FILE  = os.path.join(_CONFIG_DIR, "policies.yaml")


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Criterion:
    name:           str
    enabled:        bool
    judge_model:    str
    weight:         float         = 1.0
    pass_threshold: Optional[float] = None  # None → inherit from profile
    fail_threshold: Optional[float] = None  # None → inherit from profile

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
class Policy:
    name:         str
    description:  str
    eval_profile: str
    severity:     Optional[str]   = None   # low / medium / high
    keywords:     list[str]       = field(default_factory=list)


# ── ProfileLoader ─────────────────────────────────────────────────────────────

class ProfileLoader:
    """
    Loads evaluation_profiles.yaml and policies.yaml.

    Usage:
        loader = ProfileLoader()
        profile, policy_name = loader.resolve_profile(case)
        judge = Judge(profile)
        result = judge.grade(case, matched_rule=policy_name)
    """

    def __init__(
        self,
        profiles_path: str = _PROFILES_FILE,
        policies_path:  str = _POLICIES_FILE,
    ):
        with open(profiles_path, encoding="utf-8") as f:
            raw_profiles = yaml.safe_load(f)

        with open(policies_path, encoding="utf-8") as f:
            raw_policies = yaml.safe_load(f)

        self.profiles:        dict[str, Profile] = self._parse_profiles(raw_profiles)
        self.policies:        list[Policy]        = self._parse_policies(raw_policies)
        self.default_profile: str                 = raw_policies.get("default_profile", "standard")

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
