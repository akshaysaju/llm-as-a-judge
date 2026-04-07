"""
run.py — Complaint LLM-as-a-Judge Workflow (Profile-Driven)
──────────────────────────────────────────────────────────
Run: python run.py

For each case:
  1. Resolves the evaluation profile from business metadata via ProfileLoader
  2. Instantiates a Judge with that profile
  3. Grades the case and prints per-criterion scores with weights
  4. Applies verdict using profile-level + per-criterion thresholds
  5. Saves a full audit log to detailed_report.txt
"""

import sys
import os
from collections import Counter, defaultdict

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from cases import CASES
from judge import Judge
from profile_loader import ProfileLoader, Profile

# ── Init ──────────────────────────────────────────────────────────────────────

console = Console()
loader  = ProfileLoader()

STYLE = {"PASS": "green", "NEEDS_REVIEW": "yellow", "FAIL": "red bold"}

PROFILE_COLORS = {
    "standard":      "cyan",
    "stringent":     "magenta",
    "fraud_focused": "red",
    "loan_focused":  "yellow",
}

def profile_color(name: str) -> str:
    return PROFILE_COLORS.get(name, "blue")

def fmt(v: float) -> str:
    c = "green" if v >= 0.75 else ("yellow" if v >= 0.50 else "red")
    return f"[{c}]{v:.2f}[/{c}]"

def fmt_flag(flag: str) -> str:
    return f"[red]{flag}[/red]" if flag != "ok" else f"[green]{flag}[/green]"

def fmt_weight(w: float) -> str:
    if w >= 2.0:   return f"[bold magenta]×{w:.1f}[/bold magenta]"
    if w >= 1.5:   return f"[magenta]×{w:.1f}[/magenta]"
    if w < 1.0:    return f"[dim]×{w:.1f}[/dim]"
    return f"[white]×{w:.1f}[/white]"

# ── Header ────────────────────────────────────────────────────────────────────

console.rule("[bold blue]Complaint LLM-as-a-Judge  [Decoupled Profile Edition][/bold blue]")
console.print(f"  Profiles loaded: [cyan]{', '.join(loader.profiles.keys())}[/cyan]")
console.print(f"  Cases          : [cyan]{len(CASES)}[/cyan]\n")


# ── Profile resolution & judging (group by profile to share Judge instances) ──

# First pass: resolve all profiles
case_profiles: list[tuple[dict, Profile, str]] = []  # (case, profile, rule_name)
for case in CASES:
    profile, rule = loader.resolve_profile(case)
    case_profiles.append((case, profile, rule))

# Group cases by profile name to share Judge instances (avoids rebuilding LLMs)
judges: dict[str, Judge] = {}
for _, profile, _ in case_profiles:
    if profile.name not in judges:
        judges[profile.name] = Judge(profile)

results = []

# ── Per-case grading ──────────────────────────────────────────────────────────

for case, profile, rule in case_profiles:
    pc = profile_color(profile.name)
    console.print(Panel(
        f"[bold]{case['id']}[/bold] | "
        f"Profile: [{pc}]{profile.name}[/{pc}] "
        f"[dim](rule: {rule})[/dim]\n"
        f"[dim]Dept: {case.get('department','?')}  "
        f"Severity: {case.get('severity','?')}  "
        f"Risk: {case.get('risk_level','?')}[/dim]",
        expand=False,
    ))

    judge  = judges[profile.name]
    result = judge.grade(case, matched_rule=rule)
    results.append(result)

    vc = STYLE[result["verdict"]]
    console.print(f"  Verdict: [{vc}]{result['verdict']}[/{vc}]  "
                  f"Weighted avg: {fmt(result['weighted_avg'])}  "
                  f"[dim](pass≥{profile.pass_threshold} / fail<{profile.fail_threshold})[/dim]")

    console.print(f"  Complaint : {case['complaint'][:100]}...")
    console.print(f"  Summary   : {case['scribe_summary'][:100]}...")

    # Per-criterion table
    crit_table = Table(box=box.MINIMAL, show_header=True, header_style="bold")
    crit_table.add_column("Criterion",  style="bold")
    crit_table.add_column("Score",      justify="right")
    crit_table.add_column("Weight",     justify="right")
    crit_table.add_column("Model",      style="dim")
    crit_table.add_column("Threshold",  style="dim")
    crit_table.add_column("Override?",  justify="center")
    crit_table.add_column("Reason")

    overridden_set = set(result.get("overridden_criteria", []))
    models_used    = result.get("models_used", {})

    for crit_name, crit in profile.criteria.items():
        is_overridden = crit_name in overridden_set
        ov_tag = "[bold yellow]YES[/bold yellow]" if is_overridden else ""

        if not crit.enabled and crit_name not in overridden_set:
            crit_table.add_row(
                crit_name, "[dim]—[/dim]", "[dim]—[/dim]",
                f"[dim]{crit.judge_model}[/dim]", "[dim]disabled[/dim]",
                "", "[dim]criterion disabled in this profile[/dim]"
            )
            continue
        if not crit.enabled and crit_name in overridden_set:
            crit_table.add_row(
                crit_name, "[dim]—[/dim]", "[dim]—[/dim]",
                f"[dim]{crit.judge_model}[/dim]", "[dim]disabled[/dim]",
                ov_tag, "[yellow]disabled by case override[/yellow]"
            )
            continue

        score      = result["scores"].get(crit_name)
        reason     = result["reasons"].get(crit_name, "")
        active_model = models_used.get(crit_name, crit.judge_model)

        eff_pass = crit.effective_pass(profile.pass_threshold)
        eff_fail = crit.effective_fail(profile.fail_threshold)
        score_str = fmt(score) if score is not None else "[dim]skipped[/dim]"

        crit_table.add_row(
            crit_name,
            score_str,
            fmt_weight(crit.weight),
            active_model,
            f"p≥{eff_pass} f<{eff_fail}",
            ov_tag,
            reason[:90] if reason else "",
        )

    console.print(crit_table)


    n = result["non_llm"]
    console.print(
        f"  Non-LLM → compression: {n['compression_ratio']} "
        f"({fmt_flag(n['compression_flag'])})  "
        f"entity_recall: {fmt(n['entity_recall'])}"
    )

    if result["disabled_criteria"]:
        console.print(f"  [dim]Disabled criteria: {', '.join(result['disabled_criteria'])}[/dim]")
    console.print()

# ── Summary table ─────────────────────────────────────────────────────────────

console.rule("[bold]Results Summary[/bold]")
summary_table = Table(box=box.SIMPLE_HEAVY)
summary_table.add_column("Case")
summary_table.add_column("Profile")
summary_table.add_column("Enabled\nCriteria", justify="right")
summary_table.add_column("Wtd Avg", justify="right")
summary_table.add_column("Entity\nRecall", justify="right")
summary_table.add_column("Ratio")
summary_table.add_column("Verdict")

for r, (case, profile, rule) in zip(results, case_profiles):
    n  = r["non_llm"]
    vc = STYLE[r["verdict"]]
    pc = profile_color(profile.name)
    enabled_count = len(r["scores"])
    summary_table.add_row(
        r["id"],
        f"[{pc}]{r['profile_name']}[/{pc}]",
        str(enabled_count),
        fmt(r["weighted_avg"]),
        fmt(n["entity_recall"]),
        f"{n['compression_ratio']} {fmt_flag(n['compression_flag'])}",
        f"[{vc}]{r['verdict']}[/{vc}]",
    )
console.print(summary_table)

# ── HITL routing counts ───────────────────────────────────────────────────────

counts = Counter(r["verdict"] for r in results)
total  = len(results)
console.print(f"\n[bold]Routing[/bold]")
console.print(f"  [green]PASS[/green]         {counts['PASS']}/{total} — auto-approved")
console.print(f"  [yellow]NEEDS_REVIEW[/yellow] {counts['NEEDS_REVIEW']}/{total} — human queue")
console.print(f"  [red]FAIL[/red]         {counts['FAIL']}/{total} — escalated")
if total > 0:
    console.print(f"\n  Workload reduced by [green]{counts['PASS']/total*100:.0f}%[/green]")

# ── Profile usage breakdown ────────────────────────────────────────────────────

profile_counts = Counter(r["profile_name"] for r in results)
console.print(f"\n[bold]Profile Distribution[/bold]")
for pname, count in profile_counts.most_common():
    pc = profile_color(pname)
    console.print(f"  [{pc}]{pname}[/{pc}]: {count} case(s)")

# ── Detailed audit report ─────────────────────────────────────────────────────

report_path = os.path.join(os.path.dirname(__file__), "detailed_report.txt")
with open(report_path, "w") as f:
    f.write("COMPLAINT LLM-AS-A-JUDGE: FULL AUDIT LOG (Profile-Driven)\n")
    f.write("=" * 80 + "\n\n")

    for r, (case, profile, rule) in zip(results, case_profiles):
        f.write(f"CASE ID      : {r['id']}\n")
        f.write(f"VERDICT      : {r['verdict']}\n")
        f.write(f"PROFILE      : {r['profile_name']}  (matched rule: {r['matched_rule']})\n")
        f.write(f"WEIGHTED AVG : {r['weighted_avg']:.2f}  "
                f"(pass≥{profile.pass_threshold} / fail<{profile.fail_threshold})\n")
        f.write(f"METADATA     : severity={case.get('severity')}  "
                f"risk={case.get('risk_level')}  "
                f"dept={case.get('department')}\n")
        f.write("-" * 40 + "\n")
        f.write(f"COMPLAINT:\n{case['complaint']}\n\n")
        f.write(f"SUMMARY:\n{case['scribe_summary']}\n\n")
        f.write(f"TRACE CATEGORY: {case['trace_category']}\n\n")

        f.write("CRITERION SCORES:\n")
        for crit_name, crit in profile.criteria.items():
            eff_pass = crit.effective_pass(profile.pass_threshold)
            eff_fail = crit.effective_fail(profile.fail_threshold)
            if not crit.enabled:
                f.write(f"  - {crit_name.upper()}: DISABLED\n")
                continue
            score      = r["scores"].get(crit_name, "N/A")
            reason     = r["reasons"].get(crit_name, "")
            weight     = r["weights"].get(crit_name, 1.0)
            score_str  = f"{score:.2f}" if isinstance(score, float) else str(score)
            f.write(
                f"  - {crit_name.upper()} "
                f"(score={score_str} "
                f"weight=x{weight} model={crit.judge_model} "
                f"pass>={eff_pass} fail<{eff_fail}):\n"
                f"    {reason}\n"
            )


        f.write("\nNON-LLM METRICS:\n")
        f.write(f"  - Compression Ratio: {r['non_llm']['compression_ratio']} ({r['non_llm']['compression_flag']})\n")
        f.write(f"  - Entity Recall    : {r['non_llm']['entity_recall']}\n")

        f.write("\nRAW LLM RESPONSES (UNPARSED):\n")
        for key, raw_text in r.get("raw", {}).items():
            f.write(f"  [{key.upper()}]\n  {raw_text.strip()}\n\n")

        f.write("=" * 80 + "\n\n")

console.print(f"\n[bold blue]Done![/bold blue] Full audit saved to: [cyan]{report_path}[/cyan]")
