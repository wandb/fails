#!/usr/bin/env python3
"""
Test script for the arrow selector.
Run this directly in your terminal: python test_selector.py
"""

from simple_arrow_selector import SimpleArrowSelector
from rich.console import Console
from rich.panel import Panel


def main():
    console = Console()
    
    # Sample columns
    columns = [
        "id", "name", "status", "created_at", "updated_at",
        "user.id", "user.name", "user.email", "user.role",
        "project.id", "project.name", "project.description", 
        "metrics.views", "metrics.clicks", "metrics.conversion",
        "output.result", "output.score", "output.confidence",
        "tags", "category", "priority"
    ]
    
    preselected = {"id", "user.name", "project.name", "output.result"}
    
    # Show intro
    console.print("\n")
    console.print(Panel(
        "[bold cyan]Arrow-Based Column Selector[/bold cyan]\n\n"
        "This selector provides a clean UI for selecting items using arrow keys.\n\n"
        "Controls:\n"
        "• ↑/↓ or j/k: Navigate\n"
        "• Space: Toggle selection\n" 
        "• a: Select all\n"
        "• n: Select none\n"
        "• q: Finish and return\n\n"
        "Press Enter to start...",
        title="Instructions",
        border_style="cyan"
    ))
    
    input()  # Wait for Enter
    
    # Run selector
    selector = SimpleArrowSelector(columns, preselected)
    selected = selector.run()
    
    # Show results
    console.print("\n[bold green]Selection Complete![/bold green]\n")
    console.print(f"You selected {len(selected)} items:\n")
    
    # Group results
    grouped = {}
    other = []
    
    for col in sorted(selected):
        if '.' in col:
            prefix = col.split('.')[0]
            if prefix not in grouped:
                grouped[prefix] = []
            grouped[prefix].append(col)
        else:
            other.append(col)
    
    # Display grouped results
    for group, cols in sorted(grouped.items()):
        console.print(f"[yellow]{group}:[/yellow]")
        for col in cols:
            console.print(f"  • {col}")
        console.print()
    
    if other:
        console.print(f"[yellow]other:[/yellow]")
        for col in other:
            console.print(f"  • {col}")


if __name__ == "__main__":
    main()