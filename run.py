"""
run.py — Complaint LLM-as-a-Judge Workflow
───────────────────────────────────────────
Simulates automated ingestion of complaint cases into the HITL workflow.

  1. Load cases (as they exist after SCRIBE + TRACE have run)
  2. Judge grades each case
  3. Route to appropriate queue

Run:  python run.py
"""

from rich.console import Console
from rich.table import Table
from rich import box
from collections import Counter

from cases import CASES
from judge import Judge, JUDGE_MODEL

console = Console()

STYLE = {"PASS": "green", "NEEDS_REVIEW": "yellow", "FAIL": "red bold"}

def fmt(v):
    c = "green" if v >= 0.75 else ("yellow" if v >= 0.50 else "red")
    return f"[{c}]{v:.2f}[/{c}]"

# ─────────────────────────────────────────────────────────────────────────────

console.rule("[bold blue]Complaint LLM-as-a-Judge[/bold blue]")
console.print(f"Judge model: [cyan]{JUDGE_MODEL}[/cyan]  |  {len(CASES)} cases in manual review queue\n")

judge   = Judge()
results = []

for case in CASES:
    console.print(f"[dim]Judging {case['id']}...[/dim]", end=" ")
    result = judge.grade(case)
    results.append(result)

    vc = STYLE[result["verdict"]]
    console.print(f"[{vc}]{result['verdict']}[/{vc}]  (avg {result['avg']:.2f})")
    console.print(f"  Complaint : {case['complaint'][:90]}...")
    console.print(f"  Summary   : {case['scribe_summary'][:90]}...")
    console.print(f"  Category  : {case['trace_category']}")
    s  = result["scores"]
    i  = result["individual_scores"]
    console.print(f"  Scores (aggregated) → SCRIBE:{fmt(s['scribe_summary'])}  "
                  f"TRACE:{fmt(s['trace_category'])}  "
                  f"RESOLUTION:{fmt(s['agent_resolution'])}")
    console.print(f"  Scores (individual) →"
                  f"  accuracy:{fmt(i['scribe_accuracy'])}"
                  f"  completeness:{fmt(i['scribe_completeness'])}"
                  f"  classification:{fmt(i['trace_classification'])}"
                  f"  addressed:{fmt(i['resolution_addressed'])}"
                  f"  quality:{fmt(i['resolution_quality'])}")
    for component, note in result["notes"].items():
        if note:
            console.print(f"  [{component}] {note}")
    console.print()

# ── Summary table ─────────────────────────────────────────────────────────────

table = Table(title="Manual Review Queue — Judge Verdicts (Individual Scores)", box=box.SIMPLE_HEAVY)
table.add_column("Case")
table.add_column("Accuracy",      justify="right")
table.add_column("Completeness",  justify="right")
table.add_column("Classification",justify="right")
table.add_column("Addressed",     justify="right")
table.add_column("Quality",       justify="right")
table.add_column("Avg",           justify="right")
table.add_column("Verdict")

for r in results:
    i  = r["individual_scores"]
    vc = STYLE[r["verdict"]]
    table.add_row(
        r["id"],
        fmt(i["scribe_accuracy"]),
        fmt(i["scribe_completeness"]),
        fmt(i["trace_classification"]),
        fmt(i["resolution_addressed"]),
        fmt(i["resolution_quality"]),
        fmt(r["avg"]),
        f"[{vc}]{r['verdict']}[/{vc}]",
    )
console.print(table)

# ── HITL Routing breakdown ────────────────────────────────────────────────────

counts = Counter(r["verdict"] for r in results)
total  = len(results)
saved  = counts["PASS"]

console.print(f"\n[bold]Human-in-the-Loop Routing[/bold]")
console.print(f"  [green]PASS[/green]         {counts['PASS']}/{total} — auto-approved, removed from manual queue")
console.print(f"  [yellow]NEEDS_REVIEW[/yellow] {counts['NEEDS_REVIEW']}/{total} — sent to human reviewer")
console.print(f"  [red]FAIL[/red]         {counts['FAIL']}/{total} — escalated to senior reviewer")
console.print(f"\n  Manual review workload reduced by [green]{saved/total*100:.0f}%[/green] ({saved} of {total} cases handled by LLM Judge)")
