"""
run_consistency.py — 3-Run Consistency Test
─────────────────────────────────────────────
Runs the full LLM judge pipeline 3 times and writes every detail
(scores, reasons, raw LLM responses, verdicts) to consistency_report.txt.

Also prints a drift analysis at the end:
  - Max score swing per criterion per case across the 3 runs
  - Cases where verdict changed between runs
"""

import sys
import os
import time
from datetime import datetime
from collections import defaultdict

from rich.console import Console
from rich.table import Table
from rich.rule import Rule
from rich import box

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cases import CASES
from judge import Judge
from profile_loader import ProfileLoader

# ── Config ────────────────────────────────────────────────────────────────────

N_RUNS       = 3
REPORT_FILE  = os.path.join(os.path.dirname(__file__), "consistency_report.txt")
console      = Console()
loader       = ProfileLoader()

STYLE = {"PASS": "green", "NEEDS_REVIEW": "yellow", "FAIL": "red bold"}

def fmt(v: float) -> str:
    c = "green" if v >= 0.75 else ("yellow" if v >= 0.50 else "red")
    return f"[{c}]{v:.2f}[/{c}]"

def fmt_flag(flag: str) -> str:
    return f"[red]{flag}[/red]" if flag != "ok" else f"[green]{flag}[/green]"

# ── Resolve profiles once (same for every run) ─────────────────────────────

case_profiles = []
for case in CASES:
    profile, rule = loader.resolve_profile(case)
    case_profiles.append((case, profile, rule))

# ── Run loop ──────────────────────────────────────────────────────────────────

all_runs: list[list[dict]] = []   # all_runs[run_idx][case_idx] = result

with open(REPORT_FILE, "w") as f:

    f.write("=" * 80 + "\n")
    f.write(" COMPLAINT LLM-AS-A-JUDGE — 3-RUN CONSISTENCY REPORT\n")
    f.write(f" Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f" Models    : mistral (primary), gemma3:4b (fraud/compliance), llama3.2 (light)\n")
    f.write(f" Cases     : {len(CASES)}\n")
    f.write("=" * 80 + "\n\n")

    for run_num in range(1, N_RUNS + 1):
        console.rule(f"[bold blue]RUN {run_num} / {N_RUNS}[/bold blue]")
        f.write(f"\n{'#' * 80}\n")
        f.write(f"# RUN {run_num} of {N_RUNS}\n")
        f.write(f"{'#' * 80}\n\n")

        # Build fresh Judge instances for each run (temperature=0 but verifying repeatability)
        judges: dict[str, Judge] = {}
        for _, profile, _ in case_profiles:
            if profile.name not in judges:
                judges[profile.name] = Judge(profile)

        run_results = []
        run_start = time.time()

        for case, profile, rule in case_profiles:
            console.print(f"  [bold]{case['id']}[/bold]  profile=[cyan]{profile.name}[/cyan]  ", end="")

            t0     = time.time()
            result = judges[profile.name].grade(case, matched_rule=rule)
            elapsed = round(time.time() - t0, 1)
            run_results.append(result)

            vc = STYLE[result["verdict"]]
            console.print(
                f"verdict=[{vc}]{result['verdict']}[/{vc}]  "
                f"weighted_avg={fmt(result['weighted_avg'])}  "
                f"[dim]{elapsed}s[/dim]"
            )

            # ── Write full case detail to file ────────────────────────────────
            f.write(f"┌─ {case['id']} ─────────────────────────────────────────────────────────\n")
            f.write(f"│  Profile      : {profile.name}  (rule: {rule})\n")
            f.write(f"│  Verdict      : {result['verdict']}\n")
            f.write(f"│  Weighted Avg : {result['weighted_avg']:.4f}  "
                    f"(pass≥{profile.pass_threshold} / fail<{profile.fail_threshold})\n")
            f.write(f"│  Elapsed      : {elapsed}s\n")
            f.write(f"│  Metadata     : severity={case.get('severity')}  "
                    f"risk={case.get('risk_level')}  dept={case.get('department')}\n")
            f.write(f"│\n")
            f.write(f"│  COMPLAINT:\n")
            for line in case['complaint'].split('. '):
                f.write(f"│    {line.strip()}.\n")
            f.write(f"│\n")
            f.write(f"│  SCRIBE SUMMARY:\n")
            for line in case['scribe_summary'].split('. '):
                f.write(f"│    {line.strip()}.\n")
            f.write(f"│\n")
            f.write(f"│  TRACE CATEGORY: {case['trace_category']}\n")
            f.write(f"│\n")

            # Per-criterion scores
            f.write(f"│  CRITERION SCORES:\n")
            for crit_name, crit in profile.criteria.items():
                eff_pass = crit.effective_pass(profile.pass_threshold)
                eff_fail = crit.effective_fail(profile.fail_threshold)
                if not crit.enabled:
                    f.write(f"│    {'─'*2} {crit_name.upper():<30} DISABLED\n")
                    continue
                score  = result["scores"].get(crit_name)
                reason = result["reasons"].get(crit_name, "no reason")
                weight = result["weights"].get(crit_name, 1.0)
                score_disp  = f"{score:.2f}" if score is not None else "N/A"
                weight_disp = f"{weight}"
                flag = ""
                if score is not None and score < eff_fail:
                    flag = "  <<< BELOW FAIL THRESHOLD"
                elif score is not None and score < eff_pass:
                    flag = "  < below pass threshold"
                f.write(
                    f"| -- {crit_name.upper():<30} "
                    f"score={score_disp:<6} "
                    f"weight=x{weight_disp:<5} "
                    f"model={crit.judge_model:<14} "
                    f"pass>={eff_pass} fail<{eff_fail}"
                    f"{flag}\n"
                )
                f.write(f"|    Reason: {reason}\n")


            # Non-LLM metrics
            n = result["non_llm"]
            f.write(f"│\n")
            f.write(f"│  NON-LLM METRICS:\n")
            f.write(f"│    compression_ratio : {n['compression_ratio']}  ({n['compression_flag']})\n")
            f.write(f"│    entity_recall     : {n['entity_recall']}\n")

            # Raw LLM responses
            f.write(f"│\n")
            f.write(f"│  RAW LLM RESPONSES:\n")
            for key, raw_text in result.get("raw", {}).items():
                f.write(f"│    [{key.upper()}]:\n")
                for line in raw_text.strip().split('\n'):
                    f.write(f"│      {line}\n")
                f.write(f"│\n")

            f.write(f"└{'─' * 78}\n\n")

        run_elapsed = round(time.time() - run_start, 1)
        f.write(f"Run {run_num} total time: {run_elapsed}s\n\n")
        console.print(f"  Run {run_num} done in [cyan]{run_elapsed}s[/cyan]\n")
        all_runs.append(run_results)

    # ── Consistency Analysis ──────────────────────────────────────────────────

    f.write(f"\n{'=' * 80}\n")
    f.write(f" CONSISTENCY ANALYSIS  ({N_RUNS} runs)\n")
    f.write(f"{'=' * 80}\n\n")

    console.rule("[bold]Consistency Analysis[/bold]")

    # Build matrices: case → criterion → [score_run1, score_run2, score_run3]
    all_case_ids = [c["id"] for c in CASES]
    score_matrix: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    verdict_matrix: dict[str, list] = defaultdict(list)
    wavg_matrix: dict[str, list]    = defaultdict(list)

    for run_results in all_runs:
        for result in run_results:
            cid = result["id"]
            verdict_matrix[cid].append(result["verdict"])
            wavg_matrix[cid].append(result["weighted_avg"])
            for crit_name, score in result["scores"].items():
                score_matrix[cid][crit_name].append(score)

    # Verdict consistency
    f.write("VERDICT CONSISTENCY:\n")
    f.write(f"  {'Case':<12} {'Run1':<14} {'Run2':<14} {'Run3':<14} {'Consistent?'}\n")
    f.write(f"  {'-'*12} {'-'*14} {'-'*14} {'-'*14} {'-'*12}\n")

    verdict_table = Table(box=box.SIMPLE_HEAVY, title="Verdict Consistency")
    verdict_table.add_column("Case")
    verdict_table.add_column("Run 1")
    verdict_table.add_column("Run 2")
    verdict_table.add_column("Run 3")
    verdict_table.add_column("Weighted Avg (min→max)")
    verdict_table.add_column("Consistent?")

    for cid in all_case_ids:
        verdicts = verdict_matrix[cid]
        wavgs    = wavg_matrix[cid]
        consistent = len(set(verdicts)) == 1
        verdicts_padded = verdicts + ["—"] * (N_RUNS - len(verdicts))
        wavg_str = f"{min(wavgs):.2f} → {max(wavgs):.2f}" if wavgs else "—"

        # File
        f.write(
            f"  {cid:<12} {verdicts_padded[0]:<14} {verdicts_padded[1]:<14} "
            f"{verdicts_padded[2]:<14} {'✅ YES' if consistent else '❌ CHANGED'}\n"
        )

        # Console
        v_strs = [f"[{STYLE[v]}]{v}[/{STYLE[v]}]" for v in verdicts]
        verdict_table.add_row(
            cid,
            v_strs[0] if len(v_strs) > 0 else "—",
            v_strs[1] if len(v_strs) > 1 else "—",
            v_strs[2] if len(v_strs) > 2 else "—",
            wavg_str,
            "[green]✅ YES[/green]" if consistent else "[red]❌ CHANGED[/red]",
        )

    f.write("\n")
    console.print(verdict_table)

    # Score drift per criterion
    f.write("SCORE DRIFT PER CRITERION (max - min across 3 runs):\n\n")

    drift_table = Table(box=box.SIMPLE_HEAVY, title="Score Drift (max − min across 3 runs)")
    drift_table.add_column("Case")
    drift_table.add_column("Criterion")
    drift_table.add_column("Run1", justify="right")
    drift_table.add_column("Run2", justify="right")
    drift_table.add_column("Run3", justify="right")
    drift_table.add_column("Drift", justify="right")
    drift_table.add_column("Stability")

    high_drift_cases = []

    for cid in all_case_ids:
        for crit_name, scores in sorted(score_matrix[cid].items()):
            if len(scores) < 2:
                continue
            drift = round(max(scores) - min(scores), 2)
            scores_padded = scores + [None] * (N_RUNS - len(scores))

            def score_str(s):
                return f"{s:.2f}" if s is not None else "—"

            stability = (
                "[green]stable[/green]"   if drift <= 0.0  else
                "[green]low drift[/green]" if drift <= 0.2  else
                "[yellow]moderate[/yellow]" if drift <= 0.4  else
                "[red]HIGH DRIFT[/red]"
            )
            stability_plain = (
                "stable"      if drift <= 0.0  else
                "low drift"   if drift <= 0.2  else
                "moderate"    if drift <= 0.4  else
                "HIGH DRIFT"
            )

            if drift > 0.4:
                high_drift_cases.append((cid, crit_name, drift))

            drift_table.add_row(
                cid, crit_name,
                score_str(scores_padded[0]),
                score_str(scores_padded[1]),
                score_str(scores_padded[2]),
                f"[yellow]{drift:.2f}[/yellow]" if drift > 0 else "[green]0.00[/green]",
                stability,
            )
            f.write(
                f"  {cid:<12} {crit_name:<28} "
                f"scores=[{', '.join(score_str(s) for s in scores)}]  "
                f"drift={drift:.2f}  {stability_plain}\n"
            )

    f.write("\n")
    console.print(drift_table)

    # High drift summary
    if high_drift_cases:
        f.write("HIGH DRIFT CRITERIA (drift > 0.4):\n")
        console.print("\n[bold red]⚠  High drift criteria:[/bold red]")
        for cid, crit, drift in high_drift_cases:
            line = f"  {cid} / {crit} — drift {drift:.2f}"
            f.write(line + "\n")
            console.print(f"[red]{line}[/red]")
    else:
        f.write("No high-drift criteria detected — results are consistent.\n")
        console.print("\n[green]✅ No high-drift criteria — all runs consistent.[/green]")

    f.write(f"\nReport written at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

console.print(f"\n[bold blue]Done![/bold blue] Full report saved to: [cyan]{REPORT_FILE}[/cyan]")
