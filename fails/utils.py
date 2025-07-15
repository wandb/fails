from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import weave
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fails.prompts import Category, FinalClassificationResult


@weave.op
def display_evaluation_summary(
    eval_data: Dict[str, Any],
    failure_config: Optional[Dict[str, Any]],
    console: Console
) -> None:
    """
    Display a formatted summary of the evaluation data.
    
    Args:
        eval_data: The evaluation data dictionary
        failure_config: Optional failure filter configuration
        console: Rich console for output
    """
    op_name = eval_data["evaluation"].get("op_name", "Unknown")
    # Truncate long op names
    if len(op_name) > 100:
        op_name = op_name[:90] + "..."

    # Build evaluation info
    eval_info = f"""[bold green]Evaluation ID:[/bold green] {eval_data["evaluation"]["id"]}
[bold green]Op Name:[/bold green] {op_name}
[bold green]Total traces:[/bold green] {eval_data["trace_count"]["total"]}
[bold green]Direct children:[/bold green] {eval_data["trace_count"].get("direct_children", 0)}"""

    # If we have a failure filter, add info about filtered results
    if failure_config:
        eval_info += f"\n[bold green]Failure filter:[/bold green] {failure_config['failure_column']} == {failure_config['failure_value']}"
        eval_info += f"\n[bold green]Filtered children:[/bold green] {len(eval_data.get('children', []))}"

    console.print(Panel(eval_info, title="📊 Evaluation Summary", border_style="green"))

    # Show evaluation summary if available
    if "summary" in eval_data["evaluation"]:
        console.print(
            f"[yellow]Evaluation Summary: {eval_data['evaluation']['summary']}[/yellow]"
        )


@weave.op
def validate_failure_column(
    eval_data: Dict[str, Any],
    failure_config: Dict[str, Any],
    console: Console
) -> None:
    """
    Validate that the failure column exists and is boolean.
    
    Args:
        eval_data: The evaluation data dictionary
        failure_config: Failure filter configuration with 'failure_column' and 'failure_value'
        console: Rich console for output
        
    Raises:
        ValueError: If the failure column is invalid
    """
    if not eval_data.get("children"):
        return
        
    # Check the first child trace to validate the failure column
    first_child = eval_data["children"][0]
    
    # Navigate to the nested field
    value = first_child
    try:
        for part in failure_config["failure_column"].split("."):
            value = value.get(part, None)
            if value is None:
                break
        
        if value is None:
            console.print(f"[red]Warning: Failure column '{failure_config['failure_column']}' not found in traces![/red]")
        elif not isinstance(value, bool):
            console.print(f"[red]Error: Failure column '{failure_config['failure_column']}' is not a boolean! Value: {value} (type: {type(value).__name__})[/red]")
            raise ValueError(f"Selected failure column '{failure_config['failure_column']}' is not a boolean column")
    except (AttributeError, TypeError) as e:
        console.print(f"[red]Error accessing failure column: {e}[/red]")
        raise ValueError(f"Error accessing failure column '{failure_config['failure_column']}': {e}")


@weave.op
def prepare_trace_data_for_pipeline(
    eval_data: Dict[str, Any],
    debug: bool,
    console: Console,
    n_samples: int | None = None
) -> List[Dict[str, Any]]:
    """
    Prepare trace data from evaluation data for pipeline processing.
    
    Args:
        eval_data: The evaluation data dictionary
        debug: Whether to display debug information
        console: Rich console for output
        
    Returns:
        List of trace entries formatted for the pipeline
    """
    trace_data = []
    
    # Display child trace information
    if eval_data.get("children"):
        if n_samples:
            eval_data["children"] = eval_data["children"][:n_samples]

        console.print(
            f"[dim]First child keys:[/dim] {', '.join(eval_data['children'][0].keys())}\n"
        )
        
        if debug:
            console.print("\n[dim]CHILDREN:[/dim]")
            console.print(
                f"[dim]{len(eval_data['children'])} children found, sampling first {n_samples}:[/dim]\n"
            )
            eval_data["children"] = eval_data["children"][:n_samples]
        
        for i, trace in enumerate(eval_data["children"]):
            # Format trace entry for pipeline
            trace_entry = {
                "id": trace.get("id"),
                "inputs": trace.get("inputs", {}),
                "output": trace.get("output", {}),
                "scores": trace.get("output", {}).get("scores", {}) if trace.get("output") else {},
            }
            
            trace_data.append(trace_entry)
            
            # Display debug information for first trace
            if debug and i == 0:
                display_trace_debug_info(trace, trace_entry, i, console)
    else:
        console.print(f"[red]No children found in eval_data[/red]")
        raise ValueError("No children found in eval_data")
    
    return trace_data


def display_trace_debug_info(
    trace: Dict[str, Any],
    trace_entry: Dict[str, Any],
    index: int,
    console: Console
) -> None:
    """
    Display debug information for a trace.
    
    Args:
        trace: The original trace data
        trace_entry: The formatted trace entry
        index: The trace index
        console: Rich console for output
    """
    # Create a table for trace details
    trace_table = Table(
        title=f"Trace {index + 1} Details",
        show_header=True,
        header_style="bold magenta",
    )
    trace_table.add_column("Property", style="cyan", width=20)
    trace_table.add_column("Value", style="white")

    trace_table.add_row("ID", trace_entry.get("id", "N/A"))
    trace_table.add_row("Name", str(trace.get("display_name", "N/A")))
    trace_table.add_row("Op Name", trace.get("op_name", "N/A"))
    trace_table.add_row("Started At", trace.get("started_at", "N/A"))
    trace_table.add_row("Ended At", trace.get("ended_at", "N/A"))
    trace_table.add_row("Summary", "\n" + str(trace.get("summary", "N/A")))
    
    if (
        trace_entry.get("output")
        and isinstance(trace_entry.get("output"), dict)
        and "output" in trace_entry["output"]
    ):
        trace_table.add_row("Output", "\n" + str(trace["output"]["output"]))
    
    console.print(trace_table)
    console.print("[dim]" + "─" * 50 + "[/dim]\n")


def filter_dict_by_paths(
    data: Dict[str, Any], allowed_paths: set[str]
) -> Dict[str, Any]:
    """Filter a nested dictionary to only include data from allowed paths."""
    result = {}

    # Group paths by their top-level key
    paths_by_top_level = {}
    for path in allowed_paths:
        parts = path.split(".", 1)
        top_level = parts[0]
        if top_level not in paths_by_top_level:
            paths_by_top_level[top_level] = []
        if len(parts) > 1:
            paths_by_top_level[top_level].append(parts[1])
        else:
            # This is a top-level field
            paths_by_top_level[top_level] = None

    # Process each top-level key
    for top_level, sub_paths in paths_by_top_level.items():
        if top_level not in data:
            continue

        if sub_paths is None:
            # Include the entire top-level value
            result[top_level] = data[top_level]
        else:
            # Filter nested structure
            result[top_level] = filter_nested_dict(data[top_level], sub_paths)

    return result


def filter_nested_dict(data: Any, allowed_sub_paths: List[str]) -> Any:
    """Recursively filter nested dictionary based on allowed sub-paths."""
    if not isinstance(data, dict):
        return data

    result = {}

    # Group sub-paths by their next level
    paths_by_next_level = {}
    direct_keys = set()

    for path in allowed_sub_paths:
        parts = path.split(".", 1)
        if len(parts) == 1:
            # Direct key at this level
            direct_keys.add(parts[0])
        else:
            # Nested path
            next_level = parts[0]
            if next_level not in paths_by_next_level:
                paths_by_next_level[next_level] = []
            paths_by_next_level[next_level].append(parts[1])

    # Include direct keys
    for key in direct_keys:
        if key in data:
            result[key] = data[key]

    # Recursively filter nested structures
    for key, sub_paths in paths_by_next_level.items():
        if key in data:
            result[key] = filter_nested_dict(data[key], sub_paths)

    return result if result else None


@weave.op
def filter_trace_data_by_columns(
    traces: List[Dict[str, Any]], selected_columns: List[str]
) -> List[Dict[str, Any]]:
    """
    Filter trace data to only include the selected column paths.

    Args:
        traces: List of trace dictionaries
        selected_columns: List of column paths to keep (e.g., ["inputs.example.call_name"])

    Returns:
        List of filtered trace dictionaries
    """
    selected_paths = set(selected_columns)
    filtered_traces = []

    for trace in traces:
        filtered_trace = filter_dict_by_paths(trace, selected_paths)
        # Always preserve essential fields
        for key in ["id", "trace_id", "parent_id", "display_name"]:
            if key in trace and key not in filtered_trace:
                filtered_trace[key] = trace[key]
        filtered_traces.append(filtered_trace)

    return filtered_traces


@weave.op
def generate_evaluation_report(
    final_classification_results: List[FinalClassificationResult],
    all_categories: List[Category],
    eval_name: str,
) -> str:
    """
    Generate an evaluation report from classification results.

    Args:
        final_classification_results: List of classification results
        all_categories: List of all available categories
        eval_name: Name of the evaluation

    Returns:
        Formatted report string
    """
    # Create a summary of classifications
    classification_summary = {}
    total_failures = len(final_classification_results)

    for result in final_classification_results:
        category = result.failure_category
        if category not in classification_summary:
            classification_summary[category] = {"traces": [], "category_info": None}
        classification_summary[category]["traces"].append(
            {
                "trace_id": result.trace_id,
                "notes": result.categorization_reason,
            }
        )

    # Get category info from all_categories
    for category in all_categories:
        if category.failure_category_name in classification_summary:
            classification_summary[category.failure_category_name]["category_info"] = category

    # Sort categories by count (descending)
    sorted_categories = sorted(
        classification_summary.items(), key=lambda x: len(x[1]["traces"]), reverse=True
    )

    # Generate report
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    report = f"## {eval_name} Evaluation Report - {current_time}\n\n"

    # Add summary table to the report
    # Calculate max widths for alignment
    max_category_width = max(
        len(category_name.replace("_", " ").title())
        for category_name in classification_summary.keys()
    )
    max_category_width = max(max_category_width, len("Category"))

    # Create the table header
    report += f"{'Category'.ljust(max_category_width)} | {'Count'.center(10)} | {'Percentage'.center(12)}\n"
    report += f"{'-' * max_category_width} | {'-' * 10} | {'-' * 12}\n"

    # Add table rows
    for category_name, category_data in sorted_categories:
        traces = category_data["traces"]
        count = len(traces)
        percentage = (count / total_failures) * 100
        display_name = category_name.replace("_", " ").title()

        report += f"{display_name.ljust(max_category_width)} | {str(count).center(10)} | {f'{percentage:.1f}%'.center(12)}\n"

    report += "\n"
    report += f"### Failure Categories:\n\n"

    for idx, (category_name, category_data) in enumerate(sorted_categories, 1):
        traces = category_data["traces"]
        category_info = category_data["category_info"]
        count = len(traces)
        percentage = (count / total_failures) * 100

        # Format category name for display
        display_name = category_name.replace("_", " ").title()

        report += f"{idx}. **{display_name}**\n\n"
        report += f"Count: {count} ({percentage:.1f}% of failures)\n\n"

        if category_info:
            report += f"{category_info.failure_category_definition}\n\n"

        # Add examples section only if there are notes to show
        has_examples = any(trace["notes"] for trace in traces[:5])
        if has_examples:
            report += "Examples:\n\n"

            # Show up to 5 example trace IDs and notes
            for i, trace in enumerate(traces[:5]):
                if trace["notes"]:
                    report += f"  Trace: {trace['trace_id']}\n"
                    report += f"  {trace['notes']}\n"
                    if i < min(4, len(traces) - 1):
                        report += "\n"

        # Add list of all trace IDs for this category
        report += f"\nTrace IDs for this category:\n"
        for trace in traces:
            report += f"- {trace['trace_id']}\n"
        report += "\n"

        if idx < len(sorted_categories):
            report += "\n"

    report += "~" * 80 + "\n\n"
    report += "END REPORT\n"

    return report
