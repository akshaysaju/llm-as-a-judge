
import json
from rich.console import Console
from rich.table import Table
from cases import CASES
from judge import Judge

console = Console()

def compare_scores():
    judge = Judge()
    all_consistent = True
    
    table = Table(title="Consistency Test (Run 1 vs Run 2)")
    table.add_column("Case ID")
    table.add_column("Metric")
    table.add_column("Run 1")
    table.add_column("Run 2")
    table.add_column("Status")

    for case in CASES:
        console.print(f"Testing consistency for [bold]{case['id']}[/bold]...")
        
        # Run 1
        res1 = judge.grade(case)
        # Run 2
        res2 = judge.grade(case)
        
        metrics = list(res1["scribe"].keys()) + ["trace"]
        
        for metric in metrics:
            if metric == "trace":
                v1 = res1["trace"]["score"]
                v2 = res2["trace"]["score"]
            else:
                v1 = res1["scribe"][metric]
                v2 = res2["scribe"][metric]
            
            status = "[green]MATCH[/green]" if v1 == v2 else "[red bold]DIFF[/red bold]"
            if v1 != v2:
                all_consistent = False
            
            table.add_row(
                case["id"],
                metric,
                str(v1),
                str(v2),
                status
            )
    
    console.print(table)
    
    if all_consistent:
        console.print("\n[bold green]✅ Success: All scores are 100% consistent across both runs.[/bold green]")
    else:
        console.print("\n[bold red]⚠️ Warning: Found inconsistencies between runs.[/bold red]")

if __name__ == "__main__":
    compare_scores()
