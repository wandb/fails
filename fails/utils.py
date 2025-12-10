import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import litellm
import weave
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fails.prompts import (
    Category, 
    FinalClassificationResult,
    DEFAULT_COMPACTION_MODEL,
    TRACE_COMPACTION_SYSTEM_PROMPT,
    TRACE_COMPACTION_USER_PROMPT,
)


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
    # Build evaluation info
    eval_info = f"""[bold cyan]Evaluation ID:[/bold cyan] {eval_data["evaluation"]["id"]}"""

    # If we have a failure filter, add info about filtered results
    if failure_config:
        # Format the filter for display
        failure_filter = failure_config.get('failure_filter', {})
        if '$eq' in failure_filter:
            filter_display = f"{failure_config['failure_column']} == {failure_filter['$eq']}"
        else:
            # Handle other operators
            op_key = list(failure_filter.keys())[0] if failure_filter else ''
            op_value = failure_filter.get(op_key, '')
            op_symbols = {
                '$ne': '!=', '$gt': '>', '$gte': '>=', 
                '$lt': '<', '$lte': '<=', '$contains': 'contains',
                '$not_contains': 'does not contain', '$in': 'in', 
                '$nin': 'not in', '$exists': 'exists' if op_value else 'does not exist'
            }
            op_symbol = op_symbols.get(op_key, op_key)
            if op_key == '$exists':
                filter_display = f"{failure_config['failure_column']} {op_symbol}"
            else:
                filter_display = f"{failure_config['failure_column']} {op_symbol} {op_value}"
        
        eval_info += f"\n[bold cyan]Failure filter:[/bold cyan] {filter_display}"
        eval_info += f"\n[bold cyan]Filtered traces:[/bold cyan] {len(eval_data.get('children', []))}"

    console.print(Panel(eval_info, title="Evaluation Summary", border_style="white"))

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
    Validate that the failure column exists in the evaluation data.
    
    Args:
        eval_data: The evaluation data dictionary
        failure_config: Failure filter configuration with 'failure_column' and 'failure_value'
        console: Rich console for output
        
    Raises:
        ValueError: If the failure column doesn't exist
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
            raise ValueError(f"Selected failure column '{failure_config['failure_column']}' not found in traces")
    except (AttributeError, TypeError) as e:
        console.print(f"[red]Error accessing failure column: {e}[/red]")
        raise ValueError(f"Error accessing failure column '{failure_config['failure_column']}': {e}")


@weave.op
def prepare_trace_data_for_pipeline(
    eval_data: Dict[str, Any],
    debug: bool,
    console: Console,
    n_samples: int | None = None,
    deep_trace_analysis: bool = False,
    nesting_depth: int = 2,
    max_trace_tokens: int = 2000,
    wandb_entity: str = "",
    wandb_project: str = "",
    compaction_model: str = DEFAULT_COMPACTION_MODEL,
) -> List[Dict[str, Any]]:
    """
    Prepare trace data from evaluation data for pipeline processing.
    
    Args:
        eval_data: The evaluation data dictionary
        debug: Whether to display debug information
        console: Rich console for output
        n_samples: Maximum number of samples to process
        deep_trace_analysis: Whether to include nested trace execution trees
        nesting_depth: How deep to traverse when fetching nested traces (1=children, 2=grandchildren, etc.)
        max_trace_tokens: Token threshold for compaction
        wandb_entity: Weave entity name (required if deep_trace_analysis=True)
        wandb_project: Weave project name (required if deep_trace_analysis=True)
        compaction_model: LLM model to use for trace compaction
        
    Returns:
        List of trace entries formatted for the pipeline
    """
    trace_data = []
    
    # Display child trace information
    if eval_data.get("children"):
        if n_samples:
            eval_data["children"] = eval_data["children"][:n_samples]

        if debug:
            console.print(
                f"[dim]First child keys:[/dim] {', '.join(eval_data['children'][0].keys())}\n"
            )
            console.print("\n[dim]CHILDREN:[/dim]")
            console.print(
                f"[dim]{len(eval_data['children'])} children found, sampling first {n_samples}:[/dim]\n"
            )
            if deep_trace_analysis:
                console.print(
                    f"[dim]Deep trace analysis enabled with nesting_depth={nesting_depth}[/dim]\n"
                )
        
        for i, trace in enumerate(eval_data["children"]):
            # Format trace entry for pipeline
            trace_entry = {
                "id": trace.get("id"),
                "inputs": trace.get("inputs", {}),
                "output": trace.get("output", {}),
                "scores": trace.get("output", {}).get("scores", {}) if trace.get("output") else {},
            }
            
            # Add execution trace if deep analysis is enabled
            if deep_trace_analysis:
                if not wandb_entity or not wandb_project:
                    console.print("[yellow]Warning: deep_trace_analysis requires wandb_entity and wandb_project[/yellow]")
                else:
                    try:
                        # Fetch nested traces for this child
                        nested_traces = get_nested_traces_for_child(
                            child_trace_id=trace["id"],
                            wandb_entity=wandb_entity,
                            wandb_project=wandb_project,
                            max_depth=nesting_depth,
                        )
                        
                        if nested_traces:
                            # Format as tree
                            execution_tree = format_trace_as_tree(
                                root_trace=trace,
                                all_traces=nested_traces,
                                max_depth=nesting_depth,
                                include_inputs=True,
                                include_outputs=True,
                            )
                            
                            # Compact if too large
                            execution_tree = compact_execution_trace(
                                trace_tree=execution_tree,
                                model=compaction_model,
                                max_tokens=max_trace_tokens,
                            )
                            
                            trace_entry["execution_trace"] = execution_tree
                            
                            if debug and i == 0:
                                console.print(f"[dim]Execution trace for first child ({len(nested_traces)} nested traces):[/dim]")
                                console.print(Panel(execution_tree[:2000] + ("..." if len(execution_tree) > 2000 else ""), 
                                                   title="Execution Trace Preview", border_style="dim"))
                        else:
                            trace_entry["execution_trace"] = None
                            
                    except Exception as e:
                        console.print(f"[yellow]Warning: Failed to fetch nested traces for {trace['id']}: {e}[/yellow]")
                        trace_entry["execution_trace"] = None
            
            trace_data.append(trace_entry)
            
            # Display debug information for first trace
            if debug and i == 0:
                display_trace_debug_info(trace, trace_entry, i, console)
    else:
        console.print("[red]No children found in eval_data[/red]")
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
def generate_evaluation_report_markdown(
    final_classification_results: List[FinalClassificationResult],
    all_categories: List[Category],
    eval_name: str,
    wandb_entity: str = None,
    wandb_project: str = None,
) -> str:
    """
    Generate an evaluation report from classification results in pure Markdown format.

    Args:
        final_classification_results: List of classification results
        all_categories: List of all available categories
        eval_name: Name of the evaluation
        wandb_entity: W&B entity for generating trace URLs
        wandb_project: W&B project for generating trace URLs

    Returns:
        Formatted report string in pure Markdown
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

    # Helper function to create trace URL
    def get_trace_url(trace_id):
        if wandb_entity and wandb_project:
            return f"https://wandb.ai/{wandb_entity}/{wandb_project}/weave/calls/{trace_id}"
        return trace_id

    # Generate report in Markdown
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    report = f"# '{eval_name}' Evaluation Failures\n"
    report += f"*Generated: {current_time}*\n\n"

    # Add summary table to the report
    report += "## Summary\n\n"
    report += "| Category | Count | Percentage |\n"
    report += "|----------|-------|------------|\n"

    # Add table rows
    for category_name, category_data in sorted_categories:
        traces = category_data["traces"]
        count = len(traces)
        percentage = (count / total_failures) * 100
        display_name = category_name.replace("_", " ").title()
        
        report += f"| {display_name} | {count} | {percentage:.1f}% |\n"

    report += "\n"
    report += "## Failure Categories\n\n"

    for idx, (category_name, category_data) in enumerate(sorted_categories, 1):
        traces = category_data["traces"]
        category_info = category_data["category_info"]
        count = len(traces)
        percentage = (count / total_failures) * 100

        # Format category name for display
        display_name = category_name.replace("_", " ").title()

        report += f"### {idx}. {display_name}\n\n"
        report += f"**Count:** {count} ({percentage:.1f}% of failures)\n\n"

        if category_info:
            report += f"{category_info.failure_category_definition}\n\n"

        # Add examples section only if there are notes to show
        has_examples = any(trace["notes"] for trace in traces[:5])
        if has_examples:
            report += "**Examples:**\n\n"

            # Show up to 5 example trace IDs and notes with clickable URLs
            for i, trace in enumerate(traces[:5]):
                if trace["notes"]:
                    trace_url = get_trace_url(trace['trace_id'])
                    report += f"- **Trace:** [{trace['trace_id']}]({trace_url})\n"
                    report += f"  - {trace['notes']}\n"
                    if i < min(4, len(traces) - 1):
                        report += "\n"

        if idx < len(sorted_categories):
            report += "\n"

    # Add all trace ID lists at the end
    report += "\n## Complete Trace ID Lists by Category\n\n"
    report += "*The following sections contain all trace IDs for each failure category, which can be used for further analysis or debugging.*\n\n"
    
    for idx, (category_name, category_data) in enumerate(sorted_categories, 1):
        traces = category_data["traces"]
        display_name = category_name.replace("_", " ").title()
        
        report += f"### {idx}. {display_name} ({len(traces)} traces)\n\n"
        report += "```json\n[\n"
        trace_ids_with_links = []
        for trace in traces:
            trace_url = get_trace_url(trace['trace_id'])
            # Add the ID as a comment with URL
            trace_ids_with_links.append(f'    "{trace["trace_id"]}"  // {trace_url}')
        report += ",\n".join(trace_ids_with_links)
        report += "\n]\n```\n\n"

    report += "---\n*END REPORT*\n"

    return report


@weave.op
def generate_evaluation_report(
    final_classification_results: List[FinalClassificationResult],
    all_categories: List[Category],
    eval_name: str,
    wandb_entity: str = None,
    wandb_project: str = None,
) -> str:
    """
    Generate an evaluation report from classification results with Rich formatting for console display.

    Args:
        final_classification_results: List of classification results
        all_categories: List of all available categories
        eval_name: Name of the evaluation
        wandb_entity: W&B entity for generating trace URLs
        wandb_project: W&B project for generating trace URLs

    Returns:
        Formatted report string with Rich formatting
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

    # Helper function to create trace URL
    def get_trace_url(trace_id):
        if wandb_entity and wandb_project:
            return f"https://wandb.ai/{wandb_entity}/{wandb_project}/weave/calls/{trace_id}"
        return trace_id

    # Generate report
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    report = f"[bold bright_cyan]## '{eval_name}' Evaluation Failures[/bold bright_cyan] [dim]- {current_time}[/dim]\n\n"

    # Add summary table to the report
    # Calculate max widths for alignment
    max_category_width = max(
        len(category_name.replace("_", " ").title())
        for category_name in classification_summary.keys()
    )
    max_category_width = max(max_category_width, len("Category"))

    # Create the table header
    report += f"[bold cyan]{'Category'.ljust(max_category_width)} | {'Count'.center(10)} | {'Percentage'.center(12)}[/bold cyan]\n"
    report += f"[dim]{'-' * max_category_width} | {'-' * 10} | {'-' * 12}[/dim]\n"

    # Add table rows
    for category_name, category_data in sorted_categories:
        traces = category_data["traces"]
        count = len(traces)
        percentage = (count / total_failures) * 100
        display_name = category_name.replace("_", " ").title()

        # Color the count based on percentage (higher percentages in brighter colors)
        if percentage >= 30:
            count_color = "bright_magenta"
        elif percentage >= 10:
            count_color = "yellow"
        else:
            count_color = "white"
        
        report += f"{display_name.ljust(max_category_width)} | [{count_color}]{str(count).center(10)}[/{count_color}] | {f'{percentage:.1f}%'.center(12)}\n"

    report += "\n"
    report += "[bold bright_cyan]### Failure Categories:[/bold bright_cyan]\n\n"

    for idx, (category_name, category_data) in enumerate(sorted_categories, 1):
        traces = category_data["traces"]
        category_info = category_data["category_info"]
        count = len(traces)
        percentage = (count / total_failures) * 100

        # Format category name for display
        display_name = category_name.replace("_", " ").title()

        report += f"[bold bright_cyan]{idx}.[/bold bright_cyan] [bold bright_magenta]{display_name}[/bold bright_magenta]\n\n"
        report += f"[cyan]Count:[/cyan] {count} ({percentage:.1f}% of failures)\n\n"

        if category_info:
            report += f"{category_info.failure_category_definition}\n\n"

        # Add examples section only if there are notes to show
        has_examples = any(trace["notes"] for trace in traces[:5])
        if has_examples:
            report += "[cyan]Examples:[/cyan]\n\n"

            # Show up to 5 example trace IDs and notes with clickable URLs
            for i, trace in enumerate(traces[:5]):
                if trace["notes"]:
                    trace_url = get_trace_url(trace['trace_id'])
                    report += f"  [dim]Trace:[/dim] [link={trace_url}]{trace['trace_id']}[/link]\n"
                    report += f"  [dim]{trace['notes']}[/dim]\n"
                    if i < min(4, len(traces) - 1):
                        report += "\n"

        if idx < len(sorted_categories):
            report += "\n"

    # Add all trace ID lists at the end
    report += "\n[bold bright_cyan]### Complete Trace ID Lists by Category:[/bold bright_cyan]\n\n"
    report += "[dim]The following sections contain all trace IDs for each failure category, which can be used for further analysis or debugging.[/dim]\n\n"
    
    for idx, (category_name, category_data) in enumerate(sorted_categories, 1):
        traces = category_data["traces"]
        display_name = category_name.replace("_", " ").title()
        
        report += f"[bold bright_magenta]{idx}. {display_name}[/bold bright_magenta] ({len(traces)} traces)\n"
        report += "[dim][\n"
        trace_ids_with_links = []
        for trace in traces:
            trace_url = get_trace_url(trace['trace_id'])
            # Add the clickable ID to list
            trace_ids_with_links.append(f'    "[link={trace_url}]{trace["trace_id"]}[/link]"')
        report += ",\n".join(trace_ids_with_links)
        report += "\n][/dim]\n\n"

    report += "[bold bright_magenta]END REPORT[/bold bright_magenta]\n"

    return report


# =============================================================================
# Deep Trace Analysis Utilities
# =============================================================================

def estimate_token_count(text: str) -> int:
    """
    Estimate token count using character-based heuristic.
    
    Uses rough approximation: 4 characters ≈ 1 token, plus 10% overhead.
    
    Args:
        text: The text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    base_estimate = len(text) / 4
    with_overhead = base_estimate * 1.1
    return int(with_overhead)


def get_op_short_name(op_name: str) -> str:
    """
    Extract short name from full op_name like 'weave:///entity/project/op/Name:hash'.
    
    Args:
        op_name: Full Weave op name URI
        
    Returns:
        Short op name (e.g., "NotionRAGAgent.call_model")
    """
    if not op_name:
        return "Unknown"
    # Extract the op name part before the hash
    if "/op/" in op_name:
        name_with_hash = op_name.split("/op/")[-1]
        return name_with_hash.split(":")[0]
    return op_name


def calculate_duration(trace: Dict[str, Any]) -> str:
    """
    Calculate duration string from started_at and ended_at timestamps.
    
    Args:
        trace: Trace dictionary with started_at and ended_at fields
        
    Returns:
        Duration string (e.g., "1.13s" or "497ms")
    """
    started = trace.get("started_at")
    ended = trace.get("ended_at")
    if not started or not ended:
        return ""
    
    try:
        # Parse ISO format timestamps
        start_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(ended.replace("Z", "+00:00"))
        duration_ms = (end_dt - start_dt).total_seconds() * 1000
        
        if duration_ms >= 1000:
            return f"{duration_ms/1000:.2f}s"
        else:
            return f"{duration_ms:.0f}ms"
    except Exception:
        return ""


def _format_value_for_tree(value: Any, indent: str = "", max_length: int = 500) -> str:
    """
    Format a value for display in the tree, truncating if necessary.
    
    Args:
        value: The value to format
        indent: Current indentation string
        max_length: Maximum length before truncation
        
    Returns:
        Formatted string representation
    """
    try:
        if value is None:
            return "null"
        formatted = json.dumps(value, indent=2, default=str)
        # Add indentation to all lines
        lines = formatted.split('\n')
        if len(lines) > 1:
            formatted = lines[0] + '\n' + '\n'.join(indent + '  ' + line for line in lines[1:])
        # Truncate if too long
        if len(formatted) > max_length:
            formatted = formatted[:max_length] + "... [truncated]"
        return formatted
    except Exception:
        return str(value)[:max_length]


def format_trace_as_tree(
    root_trace: Dict[str, Any],
    all_traces: List[Dict[str, Any]],
    max_depth: Optional[int] = None,
    include_inputs: bool = True,
    include_outputs: bool = True,
) -> str:
    """
    Format traces as an ASCII tree with full details.
    
    Args:
        root_trace: The root trace to start from
        all_traces: All descendant traces (flat list)
        max_depth: Maximum depth to display (None for unlimited)
        include_inputs: Whether to include input details
        include_outputs: Whether to include output details
        
    Returns:
        ASCII tree string representation
    """
    # Create a map of parent_id -> children
    children_map: Dict[str, List[Dict[str, Any]]] = {}
    for trace in all_traces:
        parent_id = trace.get("parent_id")
        if parent_id:
            if parent_id not in children_map:
                children_map[parent_id] = []
            children_map[parent_id].append(trace)
    
    # Sort children by started_at
    for parent_id in children_map:
        children_map[parent_id].sort(key=lambda t: t.get("started_at", ""))
    
    lines = []
    
    def add_trace_to_tree(trace: Dict[str, Any], prefix: str = "", is_last: bool = True, depth: int = 0):
        """Recursively add trace and its children to the tree."""
        if max_depth is not None and depth > max_depth:
            return
            
        # Determine tree characters
        connector = "└── " if is_last else "├── "
        child_prefix = prefix + ("    " if is_last else "│   ")
        
        # Format trace header
        op_name = get_op_short_name(trace.get("op_name", ""))
        duration = calculate_duration(trace)
        
        if depth == 0:
            # Root trace - no connector
            lines.append(f"{op_name} ({duration})")
            current_prefix = ""
        else:
            lines.append(f"{prefix}{connector}{op_name} ({duration})")
            current_prefix = child_prefix
        
        # Add inputs if present and requested
        if include_inputs and trace.get("inputs"):
            inputs_str = _format_value_for_tree(trace["inputs"], current_prefix)
            lines.append(f"{current_prefix}├── inputs: {inputs_str}")
        
        # Add outputs if present and requested
        if include_outputs and trace.get("output"):
            output_str = _format_value_for_tree(trace["output"], current_prefix)
            has_children = trace["id"] in children_map
            output_connector = "├── " if has_children else "└── "
            lines.append(f"{current_prefix}{output_connector}output: {output_str}")
        
        # Add exception if present
        if trace.get("exception"):
            lines.append(f"{current_prefix}└── [ERROR] {trace['exception']}")
        
        # Process children
        children = children_map.get(trace["id"], [])
        for i, child in enumerate(children):
            is_last_child = (i == len(children) - 1)
            add_trace_to_tree(child, current_prefix, is_last_child, depth + 1)
    
    add_trace_to_tree(root_trace, depth=0)
    return "\n".join(lines)


def get_nested_traces_for_child(
    child_trace_id: str,
    wandb_entity: str,
    wandb_project: str,
    max_depth: int = 3,
    columns: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch all nested descendants for a single child trace.
    
    Args:
        child_trace_id: The trace ID to fetch descendants for
        wandb_entity: Weave entity name
        wandb_project: Weave project name
        max_depth: Maximum depth to traverse
        columns: Columns to retrieve
        
    Returns:
        List of all descendant traces
    """
    # Import here to avoid circular imports
    from fails.weave_query import WeaveQueryConfig, WeaveQueryClient
    
    if columns is None:
        columns = [
            "id", "parent_id", "trace_id", "op_name", 
            "started_at", "ended_at", "inputs", "output", "exception"
        ]
    
    config = WeaveQueryConfig(wandb_entity=wandb_entity, wandb_project=wandb_project)
    client = WeaveQueryClient(config)
    
    # Use recursive query
    result = client.query_descendants_recursive(
        parent_id=child_trace_id,
        columns=columns,
        max_depth=max_depth,
    )
    
    return result.get("traces", [])


@weave.op
def compact_execution_trace(
    trace_tree: str,
    model: str = DEFAULT_COMPACTION_MODEL,
    max_tokens: int = 2000,
) -> str:
    """
    Use LLM to intelligently summarize large execution traces.
    
    Preserves key information like tool calls, errors, and decisions
    while summarizing verbose intermediate outputs.
    
    Args:
        trace_tree: The formatted trace tree string
        model: LLM model to use for compaction (default from prompts.DEFAULT_COMPACTION_MODEL)
        max_tokens: Token threshold - if below this, no compaction
        
    Returns:
        Compacted trace tree string (or original if under threshold)
    """
    estimated_tokens = estimate_token_count(trace_tree)
    
    if estimated_tokens <= max_tokens:
        return trace_tree  # No compaction needed
    
    # Format the user prompt with the trace tree
    user_prompt = TRACE_COMPACTION_USER_PROMPT.format(trace_tree=trace_tree)

    try:
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": TRACE_COMPACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        # If compaction fails, return truncated original
        Console().print(f"[yellow]Warning: Trace compaction failed: {e}. Using truncated trace.[/yellow]")
        # Simple truncation fallback
        max_chars = max_tokens * 4
        if len(trace_tree) > max_chars:
            return trace_tree[:max_chars] + "\n... [trace truncated due to size]"
        return trace_tree
