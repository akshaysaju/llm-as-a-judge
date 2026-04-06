"""
run.py — Complaint LLM-as-a-Judge Workflow
Run:  python run.py
"""

from collections import Counter
from rich.console import Console
from rich.table import Table
from rich import box

from cases import CASES
from judge import Judge, JUDGE_MODEL

console = Console()
STYLE   = {"PASS": "green", "NEEDS_REVIEW": "yellow", "FAIL": "red bold"}

def fmt(v):
    c = "green" if v >= 0.75 else ("yellow" if v >= 0.50 else "red")
    return f"[{c}]{v:.2f}[/{c}]"

def fmt_flag(flag):
    return f"[red]{flag}[/red]" if flag != "ok" else f"[green]{flag}[/green]"

# ─────────────────────────────────────────────────────────────────────────────

console.rule("[bold blue]Complaint LLM-as-a-Judge[/bold blue]")
console.print(f"Judge: [cyan]{JUDGE_MODEL}[/cyan]  |  {len(CASES)} cases\n")

judge   = Judge()
results = []

for case in CASES:
    console.print(f"[bold]── {case['id']} ──[/bold]")
    result = judge.grade(case)
    results.append(result)

    vc = STYLE[result["verdict"]]
    console.print(f"Verdict: [{vc}]{result['verdict']}[/{vc}]")
    console.print(f"Complaint : {case['complaint'][:100]}...")
    console.print(f"Summary   : {case['scribe_summary'][:100]}...")
    console.print(f"Category  : {case['trace_category']}")

    # SCRIBE scores
    s = result["scribe"]
    r = result["scribe_reasons"]
    console.print(f"\n  [bold]SCRIBE scores[/bold] (avg: {fmt(result['scribe_avg'])})")
    console.print(f"  faithfulness        {fmt(s['faithfulness'])}   {r['faithfulness'][:80]}")
    console.print(f"  coverage            {fmt(s['coverage'])}   {r['coverage'][:80]}")
    console.print(f"  action_orientedness {fmt(s['action_orientedness'])}   {r['action_orientedness'][:80]}")
    console.print(f"  conciseness         {fmt(s['conciseness'])}   {r['conciseness'][:80]}")
    console.print(f"  clarity             {fmt(s['clarity'])}   {r['clarity'][:80]}")
    console.print(f"  urgency_tone        {fmt(s['urgency_tone'])}   {r['urgency_tone'][:80]}")
    console.print(f"  entity_preservation {fmt(s['entity_preservation'])}   {r['entity_preservation'][:80]}")

    # TRACE score
    t = result["trace"]
    console.print(f"\n  [bold]TRACE score[/bold]")
    console.print(f"  classification      {fmt(t['score'])}   {t['reason'][:80]}")

    # Non-LLM metrics
    n = result["non_llm"]
    console.print(f"\n  [bold]Non-LLM metrics[/bold]")
    console.print(f"  compression_ratio   {n['compression_ratio']}  ({fmt_flag(n['compression_flag'])})")
    console.print(f"  entity_recall       {fmt(n['entity_recall'])}")
    console.print()

# ── Summary table ─────────────────────────────────────────────────────────────

table = Table(title="Results Summary", box=box.SIMPLE_HEAVY)
table.add_column("Case")
table.add_column("Faith",  justify="right")
table.add_column("Cover",  justify="right")
table.add_column("Action", justify="right")
table.add_column("Concise",justify="right")
table.add_column("Clarity",justify="right")
table.add_column("Urgency",justify="right")
table.add_column("Entity", justify="right")
table.add_column("TRACE",  justify="right")
table.add_column("Ratio")
table.add_column("Verdict")

for r in results:
    s  = r["scribe"]
    n  = r["non_llm"]
    vc = STYLE[r["verdict"]]
    table.add_row(
        r["id"],
        fmt(s["faithfulness"]),
        fmt(s["coverage"]),
        fmt(s["action_orientedness"]),
        fmt(s["conciseness"]),
        fmt(s["clarity"]),
        fmt(s["urgency_tone"]),
        fmt(s["entity_preservation"]),
        fmt(r["trace"]["score"]),
        f"{n['compression_ratio']} {fmt_flag(n['compression_flag'])}",
        f"[{vc}]{r['verdict']}[/{vc}]",
    )
console.print(table)

# ── HITL routing ──────────────────────────────────────────────────────────────

counts = Counter(r["verdict"] for r in results)
total  = len(results)
console.print(f"\n[bold]Routing[/bold]")
console.print(f"  [green]PASS[/green]         {counts['PASS']}/{total} — auto-approved")
console.print(f"  [yellow]NEEDS_REVIEW[/yellow] {counts['NEEDS_REVIEW']}/{total} — human queue")
console.print(f"  [red]FAIL[/red]         {counts['FAIL']}/{total} — escalated")
console.print(f"\n  Workload reduced by [green]{counts['PASS']/total*100:.0f}%[/green]")

# ── Save Detailed Report ──────────────────────────────────────────────────────

with open("detailed_report.txt", "w") as f:
    f.write("COMPLAINT LLM-AS-A-JUDGE: FULL AUDIT LOG\n")
    f.write("="*80 + "\n\n")
    
    for r in results:
        case = next(c for c in CASES if c["id"] == r["id"])
        f.write(f"CASE ID: {r['id']}\n")
        f.write(f"VERDICT: {r['verdict']}\n")
        f.write("-" * 40 + "\n")
        f.write(f"COMPLAINT:\n{case['complaint']}\n\n")
        f.write(f"SUMMARY:\n{case['scribe_summary']}\n\n")
        f.write(f"TRACE CATEGORY: {case['trace_category']}\n\n")
        
        f.write("LLM REASONING (SCRIBE):\n")
        for metric, reason in r["scribe_reasons"].items():
            score = r["scribe"][metric]
            f.write(f"  - {metric.upper()} ({score:.2f}):\n    {reason}\n")
        
        f.write(f"\nLLM REASONING (TRACE):\n")
        f.write(f"  - CLASSIFICATION ({r['trace']['score']:.2f}):\n    {r['trace']['reason']}\n")

        f.write("\nRAW LLM RESPONSES (UNPARSED):\n")
        f.write(f"  [Faithfulness]\n  {r['raw']['faithfulness'].strip()}\n\n")
        f.write(f"  [Coverage]\n  {r['raw']['coverage'].strip()}\n\n")
        f.write(f"  [Action]\n  {r['raw']['raw_a' if 'raw_a' in r['raw'] else 'action'].strip()}\n\n")
        f.write(f"  [Combined]\n  {r['raw']['combined'].strip()}\n\n")
        f.write(f"  [Trace]\n  {r['raw']['trace'].strip()}\n\n")

        f.write("\nNON-LLM METRICS:\n")
        f.write(f"  - Compression Ratio: {r['non_llm']['compression_ratio']} ({r['non_llm']['compression_flag']})\n")
        f.write(f"  - Entity Recall:     {r['non_llm']['entity_recall']}\n")
        f.write("="*80 + "\n\n")

console.print(f"\n[bold blue]Done![/bold blue] Detailed untruncated report saved to: [cyan]detailed_report.txt[/cyan]")
