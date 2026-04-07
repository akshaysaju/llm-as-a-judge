"""
test_overrides.py — Per-Case Override Smoke Test (LLM)
───────────────────────────────────────────────────────
Grades COMP-001 THREE ways and compares the results to prove
case-level overrides work correctly:

  Run A — no overrides (pure standard profile)
  Run B — entity_preservation disabled + faithfulness on gemma3:4b (from cases.py)
  Run C — faithfulness disabled + trace_classification on gemma3:4b (inline override)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cases import CASES
from judge import Judge
from profile_loader import ProfileLoader

loader   = ProfileLoader()
COMP001  = next(c for c in CASES if c["id"] == "COMP-001")
profile, rule = loader.resolve_profile(COMP001)
judge    = Judge(profile)

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ── Run A: No overrides (baseline pure profile) ────────────────
section("RUN A — baseline (no overrides, pure standard profile)")
case_a = {**COMP001, "overrides": {}}      # empty overrides = pure profile
result_a = judge.grade(case_a, matched_rule=rule)

print(f"  Disabled by profile : {result_a['disabled_criteria']}")
print(f"  Overridden criteria : {result_a['overridden_criteria']}  (should be [])")
print(f"  Faithfulness model  : {result_a['models_used'].get('faithfulness')}")
print(f"  entity_preservation : {result_a['scores'].get('entity_preservation', 'NOT RUN')}")
print(f"  Weighted avg        : {result_a['weighted_avg']}  Verdict: {result_a['verdict']}")

# ── Run B: Case-level overrides from cases.py ──────────────────
section("RUN B — COMP-001 overrides (entity_preservation OFF, faithfulness→gemma3:4b)")
result_b = judge.grade(COMP001, matched_rule=rule)

print(f"  Disabled by profile : {result_b['disabled_criteria']}")
print(f"  Overridden criteria : {result_b['overridden_criteria']}  (should be ['entity_preservation','faithfulness'])")
print(f"  Faithfulness model  : {result_b['models_used'].get('faithfulness')}  (should be gemma3:4b)")
print(f"  entity_preservation : {result_b['scores'].get('entity_preservation', 'NOT RUN')}  (should be NOT RUN)")
print(f"  Weighted avg        : {result_b['weighted_avg']}  Verdict: {result_b['verdict']}")

# ── Run C: Inline override — different combination ─────────────
section("RUN C — inline override (faithfulness OFF, trace→gemma3:4b, coverage weight=3.0)")
case_c = {
    **COMP001,
    "overrides": {
        "faithfulness":       {"enabled": False},
        "trace_classification": {"judge_model": "gemma3:4b", "weight": 3.0},
        "coverage":           {"weight": 3.0, "pass_threshold": 0.85},
    }
}
result_c = judge.grade(case_c, matched_rule=rule)

print(f"  Disabled by profile : {result_c['disabled_criteria']}")
print(f"  Overridden criteria : {result_c['overridden_criteria']}")
print(f"  faithfulness        : {result_c['scores'].get('faithfulness', 'NOT RUN')}  (should be NOT RUN)")
print(f"  trace model used    : {result_c['models_used'].get('trace_classification')}  (should be gemma3:4b)")
print(f"  coverage weight     : {result_c['weights'].get('coverage')}  (should be 3.0)")
print(f"  Weighted avg        : {result_c['weighted_avg']}  Verdict: {result_c['verdict']}")

# ── Comparison summary ─────────────────────────────────────────
section("COMPARISON SUMMARY")
all_criteria = list(profile.criteria.keys())
header = f"  {'Criterion':<28} {'Run A':>8} {'Run B':>8} {'Run C':>8}"
print(header)
print("  " + "-" * (len(header) - 2))

for crit in all_criteria:
    a = result_a['scores'].get(crit, "—")
    b = result_b['scores'].get(crit, "—")
    c = result_c['scores'].get(crit, "—")
    fmt_s = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)
    changed = " *" if (a != b or b != c) else ""
    print(f"  {crit:<28} {fmt_s(a):>8} {fmt_s(b):>8} {fmt_s(c):>8}{changed}")

print(f"\n  {'weighted_avg':<28} {result_a['weighted_avg']:>8} {result_b['weighted_avg']:>8} {result_c['weighted_avg']:>8}")
print(f"  {'verdict':<28} {result_a['verdict']:>8} {result_b['verdict']:>8} {result_c['verdict']:>8}")

# ── Validation ─────────────────────────────────────────────────
section("VALIDATION")
checks = [
    ("Run A: entity_preservation ran",      result_a['scores'].get('entity_preservation') is not None),
    ("Run B: entity_preservation skipped",  result_b['scores'].get('entity_preservation') is None),
    ("Run B: faithfulness on gemma3:4b",    result_b['models_used'].get('faithfulness') == "gemma3:4b"),
    ("Run C: faithfulness skipped",         result_c['scores'].get('faithfulness') is None),
    ("Run C: trace on gemma3:4b",           result_c['models_used'].get('trace_classification') == "gemma3:4b"),
    ("Run C: coverage weight=3.0",          result_c['weights'].get('coverage') == 3.0),
    ("Profile not mutated (standard faith model still mistral)",
        profile.criteria["faithfulness"].judge_model == "mistral"),
]

all_ok = True
for label, passed in checks:
    icon = "✅" if passed else "❌"
    print(f"  {icon}  {label}")
    if not passed:
        all_ok = False

print()
if all_ok:
    print("  ✅  All override checks passed!")
else:
    print("  ❌  Some checks failed — review above.")
    sys.exit(1)
