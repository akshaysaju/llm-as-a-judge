"""
test_profiles.py — Config & Policy Unit Tests (no LLM calls)
─────────────────────────────────────────────────────────────
Validates all wiring between policies.yaml and evaluation_profiles.yaml
without making any LLM calls. Runs in <2 seconds.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from profile_loader import ProfileLoader
from cases import CASES

loader = ProfileLoader()

pass_count = 0
fail_count = 0

def check(label: str, condition: bool):
    global pass_count, fail_count
    if condition:
        print(f"  ✅  {label}")
        pass_count += 1
    else:
        print(f"  ❌  {label}")
        fail_count += 1

def section(title: str):
    print(f"\n── {title} {'─' * (60 - len(title))}")


# ── 1. Loader initialisation ──────────────────────────────────────────────────
section("1. ProfileLoader initialisation")
check("ProfileLoader initialises without error", True)
for name in ["standard", "stringent", "fraud_focused", "loan_focused"]:
    check(f"Profile '{name}' exists", name in loader.profiles)

# ── 2. Profile thresholds ─────────────────────────────────────────────────────
section("2. Profile thresholds")
cases_thresholds = [
    ("standard",      "pass_threshold", 0.75),
    ("standard",      "fail_threshold", 0.50),
    ("stringent",     "pass_threshold", 0.80),
    ("stringent",     "fail_threshold", 0.60),
    ("fraud_focused", "pass_threshold", 0.78),
    ("fraud_focused", "fail_threshold", 0.55),
    ("loan_focused",  "pass_threshold", 0.75),
    ("loan_focused",  "fail_threshold", 0.50),
]
for profile_name, field, expected in cases_thresholds:
    actual = getattr(loader.profiles[profile_name], field)
    check(f"{profile_name}: {field}={expected}", actual == expected)

# ── 3. Policy resolution ──────────────────────────────────────────────────────
section("3. Policy resolution (keyword + severity matching)")

fraud_case = {"complaint": "Two unauthorized transactions appeared on my account", "severity": "high"}
profile, rule = loader.resolve_profile(fraud_case)
check("'unauthorized' keyword → fraud_focused", profile.name == "fraud_focused")

fraud_case2 = {"complaint": "Someone committed identity theft using my card details", "severity": "medium"}
profile2, rule2 = loader.resolve_profile(fraud_case2)
check("'identity theft' keyword → fraud_focused", profile2.name == "fraud_focused")

cfpb_case = {"complaint": "This violates CFPB regulations and I will take legal action", "severity": "high"}
profile3, _ = loader.resolve_profile(cfpb_case)
check("'CFPB' + 'legal action' keywords → stringent", profile3.name == "stringent")

high_sev_case = {"complaint": "My account has been blocked for 3 days", "severity": "high"}
profile4, _ = loader.resolve_profile(high_sev_case)
check("severity=high (no keywords) → stringent", profile4.name == "stringent")

loan_case = {"complaint": "My mortgage application has been delayed for 6 weeks", "severity": "low"}
profile5, _ = loader.resolve_profile(loan_case)
check("'mortgage' keyword → loan_focused", profile5.name == "loan_focused")

low_case = {"complaint": "I was charged a $35 fee on my account", "severity": "low"}
profile6, rule6 = loader.resolve_profile(low_case)
check("No keywords + low severity → standard (default)", profile6.name == "standard")
check("No match → rule = 'default'", rule6 == "default")

no_meta_case = {"complaint": "I have a general question about my account"}
profile7, _ = loader.resolve_profile(no_meta_case)
check("No metadata at all → standard (default)", profile7.name == "standard")

# ── 4. CASES in cases.json resolve correctly ──────────────────────────────────
section("4. CASES from cases.json resolve correctly")
expected_profiles = {
    "COMP-001": "standard",
    "COMP-002": "fraud_focused",
    "COMP-003": "loan_focused",
    "COMP-004": "fraud_focused",
    "COMP-005": "standard",
    "COMP-006": "stringent",
    "COMP-007": "standard",
}
for case in CASES:
    cid = case["id"]
    if cid in expected_profiles:
        profile, rule = loader.resolve_profile(case)
        check(f"{cid} → {expected_profiles[cid]} (rule: {rule})",
              profile.name == expected_profiles[cid])

# ── 5. trace_classification is GONE from all profiles ────────────────────────
section("5. trace_classification removed (SCRIBE evaluates summaries only)")
for pname, profile in loader.profiles.items():
    check(f"{pname}: no trace_classification criterion",
          "trace_classification" not in profile.criteria)

# ── 6. Enabled / disabled criteria ───────────────────────────────────────────
section("6. Enabled / disabled criteria")
enabled_checks = [
    ("standard",      "urgency_tone",        False),
    ("standard",      "faithfulness",        True),
    ("stringent",     "urgency_tone",        True),
    ("stringent",     "faithfulness",        True),
    ("fraud_focused", "conciseness",         False),
    ("fraud_focused", "faithfulness",        True),
    ("fraud_focused", "entity_preservation", True),
    ("loan_focused",  "urgency_tone",        True),
    ("loan_focused",  "faithfulness",        True),
]
for profile_name, crit_name, expected in enabled_checks:
    actual = loader.profiles[profile_name].criteria[crit_name].enabled
    check(f"{profile_name}.{crit_name}: enabled={expected}", actual == expected)

# ── 7. Per-criterion threshold overrides ──────────────────────────────────────
section("7. Per-criterion threshold overrides")
threshold_checks = [
    ("stringent",     "faithfulness",        "effective_pass", 0.85),
    ("stringent",     "faithfulness",        "effective_fail", 0.65),
    ("stringent",     "entity_preservation", "effective_pass", 0.85),
    ("stringent",     "entity_preservation", "effective_fail", 0.65),
    ("fraud_focused", "faithfulness",        "effective_pass", 0.80),
    ("fraud_focused", "faithfulness",        "effective_fail", 0.60),
    ("fraud_focused", "entity_preservation", "effective_pass", 0.85),
    ("fraud_focused", "entity_preservation", "effective_fail", 0.65),
    ("standard",      "faithfulness",        "effective_pass", 0.75),
    ("standard",      "faithfulness",        "effective_fail", 0.50),
    ("standard",      "conciseness",         "effective_pass", 0.60),
    ("standard",      "conciseness",         "effective_fail", 0.40),
]
for profile_name, crit_name, method, expected in threshold_checks:
    profile  = loader.profiles[profile_name]
    crit     = profile.criteria[crit_name]
    actual   = getattr(crit, method)(
        profile.pass_threshold if "pass" in method else profile.fail_threshold
    )
    check(f"{profile_name}.{crit_name}: {method}={expected}", actual == expected)

# ── 8. Criterion weights ──────────────────────────────────────────────────────
section("8. Criterion weights")
weight_checks = [
    ("fraud_focused", "faithfulness",        3.0),
    ("fraud_focused", "entity_preservation", 3.0),
    ("stringent",     "faithfulness",        2.0),
    ("stringent",     "entity_preservation", 2.0),
    ("standard",      "faithfulness",        1.5),
    ("standard",      "conciseness",         0.75),
    ("loan_focused",  "coverage",            2.0),
    ("loan_focused",  "action_orientedness", 2.0),
]
for profile_name, crit_name, expected in weight_checks:
    actual = loader.profiles[profile_name].criteria[crit_name].weight
    check(f"{profile_name}.{crit_name}: weight={expected}", actual == expected)

# ── 9. Unique judge models ────────────────────────────────────────────────────
section("9. Unique judge models per profile")
model_checks = [
    ("standard",      ["llama3.2", "mistral"]),
    ("stringent",     ["gemma3:4b", "llama3.2", "mistral"]),
    ("fraud_focused", ["gemma3:4b", "llama3.2", "mistral"]),
    ("loan_focused",  ["llama3.2", "mistral"]),
]
for profile_name, expected in model_checks:
    actual = loader.profiles[profile_name].unique_models
    check(f"{profile_name}: unique models = {expected}", actual == expected)

# ── 10. Keyword matching is case-insensitive ──────────────────────────────────
section("10. Keyword matching is case-insensitive")
upper_case = {"complaint": "I AM A VICTIM OF FRAUD AND UNAUTHORIZED CHARGES", "severity": "low"}
p10, _ = loader.resolve_profile(upper_case)
check("UPPERCASE 'FRAUD' / 'UNAUTHORIZED' → fraud_focused", p10.name == "fraud_focused")

mixed_case = {"complaint": "My Mortgage Application was rejected", "severity": "low"}
p11, _ = loader.resolve_profile(mixed_case)
check("Mixed-case 'Mortgage' → loan_focused", p11.name == "loan_focused")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}\n")
total = pass_count + fail_count
if fail_count == 0:
    print(f"✅  All {total} tests passed! Config and routing are wired correctly.")
else:
    print(f"❌  {fail_count}/{total} tests FAILED — review output above.")
    sys.exit(1)
