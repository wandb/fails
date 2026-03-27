import concurrent.futures
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import litellm
import weave
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fails.prompts import (
    Category,
    FinalClassificationResult,
    TRACE_COMPACTION_SYSTEM_PROMPT,
    TRACE_COMPACTION_USER_PROMPT,
)

# Avoid circular import - WeaveQueryClient/Config are only needed at runtime in one function
if TYPE_CHECKING:
    from fails.weave_query import WeaveQueryClient, WeaveQueryConfig


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
    # Build evaluation info - handle both evaluation and trace-only queries
    if "evaluation" in eval_data:
        eval_info = f"""[bold cyan]Evaluation ID:[/bold cyan] {eval_data["evaluation"]["id"]}"""
    else:
        # For trace-only queries (from trace URLs with filters)
        eval_info = f"""[bold cyan]Traces:[/bold cyan] {len(eval_data.get('children', []))} traces retrieved"""

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

    # Update title based on whether we have an evaluation or just traces
    title = "Evaluation Summary" if "evaluation" in eval_data else "Trace Query Summary"
    console.print(Panel(eval_info, title=title, border_style="white"))

    # Show evaluation summary if available (only for evaluation queries)
    if "evaluation" in eval_data and "summary" in eval_data["evaluation"]:
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
def extract_human_annotations(trace: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract human annotations from a trace.

    Checks common annotation locations:
    - attributes.weave.user_feedback
    - attributes.annotation
    - annotation
    - feedback
    - summary.weave.status (for annotation status)

    Args:
        trace: The trace data

    Returns:
        Dictionary of annotations found, empty if none
    """
    annotations = {}

    # Check attributes.weave for user_feedback and annotations
    if "attributes" in trace and "weave" in trace["attributes"]:
        weave_attrs = trace["attributes"]["weave"]

        if "user_feedback" in weave_attrs:
            annotations["user_feedback"] = weave_attrs["user_feedback"]

        # Check for annotation-related fields in attributes.weave
        for key in weave_attrs.keys():
            if "annotation" in key.lower() or "feedback" in key.lower():
                annotations[f"weave.{key}"] = weave_attrs[key]

    # Check attributes for annotation field
    if trace.get("attributes", {}).get("annotation"):
        annotations["annotation"] = trace["attributes"]["annotation"]

    # Check top-level annotation field
    if trace.get("annotation"):
        annotations["annotation"] = trace["annotation"]

    # Check top-level feedback field
    if trace.get("feedback"):
        annotations["feedback"] = trace["feedback"]

    # Check summary for weave status (annotated/reviewed status)
    if trace.get("summary", {}).get("weave", {}).get("status"):
        annotations["weave_status"] = trace["summary"]["weave"]["status"]

    return annotations


@weave.op
def extract_metadata(trace: Dict[str, Any], selected_columns: List[str]) -> Dict[str, Any]:
    """
    Extract metadata from a trace including scores, timestamps, and other fields.

    Args:
        trace: The trace data
        selected_columns: List of column paths to extract

    Returns:
        Dictionary of metadata
    """
    metadata = {}

    # Extract scores if they exist (evaluation-style traces)
    output = trace.get("output")
    if isinstance(output, dict) and output.get("scores"):
        metadata["scores"] = output["scores"]

    # Extract timestamps
    if trace.get("started_at"):
        metadata["started_at"] = trace["started_at"]
    if trace.get("ended_at"):
        metadata["ended_at"] = trace["ended_at"]

    # Extract summary if available
    if trace.get("summary"):
        metadata["summary"] = trace["summary"]

    # Extract exception if any
    if trace.get("exception"):
        metadata["exception"] = trace["exception"]

    # Extract any other selected columns that aren't inputs/output
    for col in selected_columns:
        if col not in ["inputs", "output", "id"] and "." in col:
            # Handle nested paths
            parts = col.split(".")
            value = trace
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    value = None
                    break
            if value is not None:
                metadata[col] = value

    return metadata


@weave.op
def prepare_trace_data_for_pipeline(
    eval_data: Dict[str, Any],
    debug: bool,
    console: Console,
    deep_trace_analysis: bool,
    compaction_model: str,
    nesting_depth: int,
    max_trace_tokens: int,
    n_samples: int | None = None,
    selected_columns: List[str] | None = None,
    wandb_entity: str = "",
    wandb_project: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Prepare trace data for pipeline processing.

    Args:
        eval_data: The evaluation data dictionary (with 'children' key)
        debug: Whether to display debug information
        console: Rich console for output
        n_samples: Optional limit on number of samples
        selected_columns: List of column paths to extract as metadata
        deep_trace_analysis: Whether to fetch and include nested execution traces
        nesting_depth: How deep to traverse when fetching nested traces
        max_trace_tokens: Token threshold before triggering compaction
        compaction_model: LLM model for trace compaction
        wandb_entity: W&B entity (required for deep_trace_analysis)
        wandb_project: W&B project (required for deep_trace_analysis)

    Returns:
        Tuple of (trace_data, annotation_summary) where:
        - trace_data: List of trace entries formatted for the pipeline
        - annotation_summary: Dict with keys 'has_annotations' and 'examples'
    """
    trace_data = []
    annotation_examples = []

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

        for i, trace in enumerate(eval_data["children"]):
            # Extract human annotations
            annotations = extract_human_annotations(trace)

            # Extract metadata (scores, timestamps, etc.)
            metadata = extract_metadata(trace, selected_columns or [])

            # Format trace entry for pipeline
            trace_entry = {
                "id": trace.get("id"),
                "inputs": trace.get("inputs", {}),
                "output": trace.get("output", {}),
                "metadata": metadata,
                "annotations": annotations,
                "execution_trace": None,  # Default to None, filled later if deep analysis enabled
            }

            trace_data.append(trace_entry)

            # Collect annotation examples for the prompt
            if annotations:
                annotation_examples.append({
                    "trace_id": trace.get("id"),
                    "annotations": annotations
                })

            # Display debug information for first trace
            if debug and i == 0:
                display_trace_debug_info(trace, trace_entry, i, console)

                # Also show annotation check details for first trace
                if annotations:
                    console.print(f"\n[green]✓ First trace has annotations:[/green]")
                    for key, value in annotations.items():
                        console.print(f"  [dim]{key}:[/dim] {str(value)[:100]}...")
                else:
                    console.print(f"\n[yellow]First trace has no annotations[/yellow]")
                    console.print(f"[dim]  Available trace keys: {list(trace.keys())[:10]}[/dim]")
                    if "attributes" in trace:
                        console.print(f"[dim]  attributes keys: {list(trace.get('attributes', {}).keys())}[/dim]")

        # Process deep trace analysis in parallel if enabled
        if deep_trace_analysis:
            if not wandb_entity or not wandb_project:
                console.print("[yellow]Warning: deep_trace_analysis requires wandb_entity and wandb_project[/yellow]")
            else:
                if debug:
                    console.print(f"[dim]Starting parallel deep trace analysis for {len(trace_data)} traces...[/dim]")
                
                # Import here to avoid circular import
                from fails.weave_query import WeaveQueryClient, WeaveQueryConfig
                
                # Initialize Weave client
                weave_config = WeaveQueryConfig(
                    wandb_entity=wandb_entity,
                    wandb_project=wandb_project,
                    api_key=os.environ.get("WANDB_API_KEY")
                )
                weave_client = WeaveQueryClient(weave_config)

                for i, entry in enumerate(trace_data):
                    try:
                        if debug:
                            console.print(f"[dim]Processing trace {i+1}/{len(trace_data)}: {entry['id']}[/dim]")
                        
                        execution_trace = process_single_trace_for_deep_analysis(
                            entry,
                            weave_client,
                            nesting_depth,
                            compaction_model,
                            max_trace_tokens,
                        )
                        entry["execution_trace"] = execution_trace
                    except Exception as e:
                        entry["execution_trace"] = None
                        if debug:
                            console.print(f"[yellow]Error processing trace {entry['id']}: {e}[/yellow]")

                if debug:
                    # Show preview of first trace's execution trace
                    if trace_data and trace_data[0].get("execution_trace"):
                        preview = trace_data[0]["execution_trace"][:1500] + ("..." if len(trace_data[0]["execution_trace"]) > 1500 else "")
                        console.print(f"\n[dim]Execution trace for first trace:[/dim]")
                        console.print(Panel(preview, title="Execution Trace Preview", border_style="dim"))
    else:
        console.print("[red]No children found in eval_data[/red]")
        raise ValueError("No children found in eval_data")


    annotation_summary = {
        "has_annotations": len(annotation_examples) > 0,
        "examples": annotation_examples[:5]  # Limit to first 5 examples for prompt
    }

    if debug:
        if annotation_summary["has_annotations"]:
            console.print(f"\n[bright_cyan]✓ Found {len(annotation_examples)} traces with human annotations[/bright_cyan]")
            console.print(f"[dim]  Annotation examples will be included in prompts[/dim]")
            # Show a sample of what annotations look like
            if annotation_examples:
                console.print(f"[dim]  Sample annotation keys: {list(annotation_examples[0]['annotations'].keys())}[/dim]")
        else:
            console.print(f"\n[yellow]No human annotations found in traces[/yellow]")
            console.print(f"[dim]  Checked for: attributes.weave.user_feedback, attributes.annotation, annotation, feedback[/dim]")

    return trace_data, annotation_summary


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
    Generate a trace pattern report from classification results in pure Markdown format.

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
    total_traces = len(final_classification_results)

    for result in final_classification_results:
        category = result.pattern_category
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
        if category.pattern_category_name in classification_summary:
            classification_summary[category.pattern_category_name]["category_info"] = category

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

    report = f"# '{eval_name}' Trace Pattern Analysis\n"
    report += f"*Generated: {current_time}*\n\n"

    # Add summary table to the report
    report += "## Summary\n\n"
    report += "| Category | Count | Percentage |\n"
    report += "|----------|-------|------------|\n"

    # Add table rows
    for category_name, category_data in sorted_categories:
        traces = category_data["traces"]
        count = len(traces)
        percentage = (count / total_traces) * 100
        display_name = category_name.replace("_", " ").title()

        report += f"| {display_name} | {count} | {percentage:.1f}% |\n"

    report += "\n"
    report += "## Pattern Categories\n\n"

    for idx, (category_name, category_data) in enumerate(sorted_categories, 1):
        traces = category_data["traces"]
        category_info = category_data["category_info"]
        count = len(traces)
        percentage = (count / total_traces) * 100

        # Format category name for display
        display_name = category_name.replace("_", " ").title()

        report += f"### {idx}. {display_name}\n\n"
        report += f"**Count:** {count} ({percentage:.1f}% of traces)\n\n"

        if category_info:
            report += f"{category_info.pattern_category_definition}\n\n"

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
    Generate a trace pattern report from classification results with Rich formatting for console display.

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
    total_traces = len(final_classification_results)

    for result in final_classification_results:
        category = result.pattern_category
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
        if category.pattern_category_name in classification_summary:
            classification_summary[category.pattern_category_name]["category_info"] = category

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

    report = f"[bold bright_cyan]## '{eval_name}' Trace Pattern Analysis[/bold bright_cyan] [dim]- {current_time}[/dim]\n\n"

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
        percentage = (count / total_traces) * 100
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
    report += "[bold bright_cyan]### Pattern Categories:[/bold bright_cyan]\n\n"

    for idx, (category_name, category_data) in enumerate(sorted_categories, 1):
        traces = category_data["traces"]
        category_info = category_data["category_info"]
        count = len(traces)
        percentage = (count / total_traces) * 100

        # Format category name for display
        display_name = category_name.replace("_", " ").title()

        report += f"[bold bright_cyan]{idx}.[/bold bright_cyan] [bold bright_magenta]{display_name}[/bold bright_magenta]\n\n"
        report += f"[cyan]Count:[/cyan] {count} ({percentage:.1f}% of traces)\n\n"

        if category_info:
            report += f"{category_info.pattern_category_definition}\n\n"

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

@weave.op
def process_single_trace_for_deep_analysis(
    trace_entry: Dict[str, Any],
    weave_client: Any,
    nesting_depth: int,
    compaction_model: str,
    max_trace_tokens: int,
) -> str | None:
    """
    Process a single trace for deep analysis - fetches descendants, formats tree, compacts.
    Extracted as separate function to enable weave tracing.
    """
    try:
        trace_id = trace_entry["id"]
        
        # Fetch descendants using Weave client
        descendants_result = weave_client.query_descendants_recursive(
            parent_id=trace_id,
            max_depth=nesting_depth,
        )
        
        all_traces = descendants_result["traces"]
        if not all_traces:
            return None
            
        # Find root trace in the fetched traces or use entry
        root_trace = next((t for t in all_traces if t["id"] == trace_id), None)
        if not root_trace:
            # Reconstruct root from trace_entry if not found
            root_trace = {
                "id": trace_entry["id"],
                "op_name": "Root",
                "inputs": trace_entry["inputs"],
                "output": trace_entry["output"],
            }
            all_traces.append(root_trace)

        # Format as tree
        execution_tree = format_trace_as_tree(
            root_trace=root_trace,
            all_traces=all_traces,
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
        
        return execution_tree
        
    except Exception as e:
        return None


@weave.op
def estimate_token_count(text: str) -> int:
    """
    Quick heuristic for token estimation (~4 chars per token).
    
    Args:
        text: The text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    return len(text) // 4


@weave.op
def get_op_short_name(op_name: str) -> str:
    """
    Extract a clean operation name from the full Weave op_name.
    
    Args:
        op_name: Full Weave op_name (e.g., "weave:///entity/project/op/MyAgent.call:abc123")
        
    Returns:
        Short name (e.g., "MyAgent.call")
    """
    if not op_name:
        return "Unknown"
    
    # Extract from weave:///entity/project/op/NAME:hash format
    if "/op/" in op_name:
        name = op_name.split("/op/")[-1]
        # Remove the hash suffix if present
        if ":" in name:
            name = name.split(":")[0]
        return name
    
    return op_name


@weave.op
def calculate_duration(trace: Dict[str, Any]) -> str:
    """
    Calculate and format the duration of a trace.
    
    Args:
        trace: Trace dictionary with started_at and ended_at fields
        
    Returns:
        Formatted duration string (e.g., "1.23s", "456ms")
    """
    started = trace.get("started_at")
    ended = trace.get("ended_at")
    
    if not started or not ended:
        return ""
    
    try:
        # Parse ISO format timestamps
        if isinstance(started, str):
            start_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
        else:
            start_dt = started
            
        if isinstance(ended, str):
            end_dt = datetime.fromisoformat(ended.replace("Z", "+00:00"))
        else:
            end_dt = ended
        
        duration_ms = (end_dt - start_dt).total_seconds() * 1000
        
        if duration_ms >= 1000:
            return f"{duration_ms / 1000:.2f}s"
        else:
            return f"{duration_ms:.0f}ms"
    except Exception:
        return ""


def _clean_value_for_tree(value: Any, seen_tools: set = None, seen_system_prompt: list = None) -> Any:
    """
    Clean a value before formatting for the trace tree.
    Removes noise and deduplicates repeated content.
    
    Args:
        value: The value to clean
        seen_tools: Set to track if we've seen tool definitions (pass same set across calls)
        seen_system_prompt: List to track if we've seen system prompt (pass same list across calls)
        
    Returns:
        Cleaned value
    """
    if seen_tools is None:
        seen_tools = set()
    if seen_system_prompt is None:
        seen_system_prompt = []
    
    if value is None:
        return None
    
    if isinstance(value, str):
        # Shorten weave URIs
        if value.startswith("weave:///"):
            # Extract just the object/op name
            # weave:///entity/project/object/Name:hash -> Name
            parts = value.split("/")
            if len(parts) >= 5:
                name_part = parts[-1].split(":")[0]
                return f"[{name_part}]"
        return value
    
    if isinstance(value, (int, float, bool)):
        return value
    
    if isinstance(value, dict):
        cleaned = {}
        for k, v in value.items():
            # Skip Weave internal metadata
            if k in ("_type", "_class_name", "_bases"):
                continue
            # Skip assistant_message if we have tool_calls (redundant)
            if k == "assistant_message" and "tool_calls" in value:
                continue
            cleaned[k] = _clean_value_for_tree(v, seen_tools, seen_system_prompt)
        return cleaned
    
    if isinstance(value, list):
        # Check if this is a tools array (tool definitions)
        if value and isinstance(value[0], dict) and value[0].get("type") == "function":
            tool_count = len(value)
            tool_sig = f"tools_{tool_count}"
            if tool_sig in seen_tools:
                return f"[{tool_count} tools - same as above]"
            seen_tools.add(tool_sig)
            # Return simplified tool list
            return [{"tool": t.get("function", {}).get("name", "unknown")} for t in value]
        
        # Check if this is a messages array
        if value and isinstance(value[0], dict) and value[0].get("role"):
            cleaned_messages = []
            for msg in value:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                # Deduplicate system prompts (same prompt repeats in every LLM call)
                # Keep FIRST one in full (tells us what agent should do)
                # Dedupe subsequent ones (they're identical)
                if role == "system":
                    if seen_system_prompt:
                        cleaned_messages.append({"role": "system", "content": "[system prompt - same as root]"})
                        continue
                    seen_system_prompt.append(True)
                    # Keep full system prompt - it defines expected behavior!
                
                # For assistant messages with tool calls, show the calls clearly
                if role == "assistant" and msg.get("tool_calls"):
                    tool_calls_info = []
                    for tc in msg.get("tool_calls", []):
                        func = tc.get("function", {})
                        tool_calls_info.append({
                            "name": func.get("name", "?"),
                            "args": func.get("arguments", "")  # Keep args - important for analysis!
                        })
                    cleaned_messages.append({
                        "role": "assistant",
                        "tool_calls": tool_calls_info
                    })
                    continue
                
                # For tool results - KEEP FULL CONTENT (critical for failure analysis!)
                # Errors, unexpected results, etc. are exactly what we need to see
                if role == "tool":
                    cleaned_messages.append({
                        "role": "tool",
                        "name": msg.get("name", ""),
                        "content": content  # Don't truncate - this is where failures show!
                    })
                    continue
                
                # User messages - can truncate, initial query is less critical
                cleaned_messages.append({
                    "role": role,
                    "content": content[:2000] + "..." if len(content) > 2000 else content
                })
            
            # Keep full message history - let compaction handle if too large
            return cleaned_messages
        
        # Regular list
        return [_clean_value_for_tree(item, seen_tools, seen_system_prompt) for item in value]
    
    return value


def _format_value_for_tree(value: Any, indent: str = "", clean: bool = True, 
                           seen_tools: set = None, seen_system_prompt: list = None) -> str:
    """
    Format a value for display in the trace tree.
    
    Args:
        value: The value to format
        indent: Current indentation
        clean: Whether to clean the value first (remove noise)
        seen_tools: Set to track tool definitions seen
        seen_system_prompt: List to track system prompts seen
        
    Returns:
        Formatted string representation
    """
    # Clean the value first to remove noise
    if clean:
        value = _clean_value_for_tree(value, seen_tools, seen_system_prompt)
    
    if value is None:
        return "null"
    
    if isinstance(value, str):
        return f'"{value}"'
    
    if isinstance(value, (int, float, bool)):
        return str(value)
    
    if isinstance(value, dict):
        if not value:
            return "{}"
        try:
            return json.dumps(value, indent=2, default=str)
        except Exception:
            return str(value)
    
    if isinstance(value, list):
        if not value:
            return "[]"
        try:
            return json.dumps(value, indent=2, default=str)
        except Exception:
            return str(value)
    
    return str(value)


@weave.op
def format_trace_as_tree(
    root_trace: Dict[str, Any],
    all_traces: List[Dict[str, Any]],
    max_depth: Optional[int] = None,
    include_inputs: bool = True,
    include_outputs: bool = True,
) -> str:
    """
    Format a list of traces as an ASCII tree structure.
    
    Automatically cleans traces to remove noise:
    - Weave internal metadata (_type, _class_name, _bases)
    - Long weave:/// URIs shortened to object names
    - Repeated tool definitions deduplicated
    - Repeated system prompts deduplicated
    
    Args:
        root_trace: The root trace to start from
        all_traces: All traces (including root and descendants)
        max_depth: Maximum depth to display (None for unlimited)
        include_inputs: Whether to include input details
        include_outputs: Whether to include output details
        
    Returns:
        ASCII tree representation of the trace hierarchy
    """
    # Build parent->children mapping
    children_map: Dict[str, List[Dict[str, Any]]] = {}
    trace_by_id: Dict[str, Dict[str, Any]] = {}
    
    for trace in all_traces:
        trace_id = trace.get("id")
        parent_id = trace.get("parent_id")
        
        if trace_id:
            trace_by_id[trace_id] = trace
            
        if parent_id:
            if parent_id not in children_map:
                children_map[parent_id] = []
            children_map[parent_id].append(trace)
    
    # Sort children by started_at
    for parent_id in children_map:
        children_map[parent_id].sort(
            key=lambda t: t.get("started_at", ""),
        )
    
    lines: List[str] = []
    
    # Shared state for deduplication across the entire tree
    seen_tools: set = set()
    seen_system_prompt: list = []
    
    def add_trace_to_tree(trace: Dict[str, Any], prefix: str = "", is_last: bool = True, depth: int = 0):
        if max_depth is not None and depth > max_depth:
            return
        
        trace_id = trace.get("id", "")
        op_name = get_op_short_name(trace.get("op_name", ""))
        duration = calculate_duration(trace)
        
        # Build the main line
        connector = "└── " if is_last else "├── "
        duration_str = f" ({duration})" if duration else ""
        lines.append(f"{prefix}{connector}{op_name}{duration_str}")
        
        # Prepare prefix for children
        child_prefix = prefix + ("    " if is_last else "│   ")
        
        # Add inputs if requested
        if include_inputs and trace.get("inputs"):
            inputs = trace.get("inputs", {})
            if inputs:
                formatted_inputs = _format_value_for_tree(
                    inputs, child_prefix, clean=True, 
                    seen_tools=seen_tools, seen_system_prompt=seen_system_prompt
                )
                if "\n" in formatted_inputs:
                    lines.append(f"{child_prefix}├── inputs:")
                    for input_line in formatted_inputs.split("\n"):
                        lines.append(f"{child_prefix}│   {input_line}")
                else:
                    lines.append(f"{child_prefix}├── inputs: {formatted_inputs}")
        
        # Add outputs if requested
        if include_outputs and trace.get("output"):
            output = trace.get("output")
            if output:
                formatted_output = _format_value_for_tree(
                    output, child_prefix, clean=True,
                    seen_tools=seen_tools, seen_system_prompt=seen_system_prompt
                )
                if "\n" in formatted_output:
                    lines.append(f"{child_prefix}├── output:")
                    for output_line in formatted_output.split("\n"):
                        lines.append(f"{child_prefix}│   {output_line}")
                else:
                    lines.append(f"{child_prefix}├── output: {formatted_output}")
        
        # Add exception if present
        if trace.get("exception"):
            lines.append(f"{child_prefix}├── [ERROR] {trace.get('exception')[:200]}")
        
        # Recurse into children
        children = children_map.get(trace_id, [])
        for i, child in enumerate(children):
            is_last_child = (i == len(children) - 1)
            add_trace_to_tree(child, child_prefix, is_last_child, depth + 1)
    
    # Start with root trace header
    root_op = get_op_short_name(root_trace.get("op_name", ""))
    root_duration = calculate_duration(root_trace)
    duration_str = f" ({root_duration})" if root_duration else ""
    lines.append(f"{root_op}{duration_str}")
    
    # Add root's children
    root_id = root_trace.get("id", "")
    children = children_map.get(root_id, [])
    for i, child in enumerate(children):
        is_last_child = (i == len(children) - 1)
        add_trace_to_tree(child, "", is_last_child, 1)
    
    return "\n".join(lines)


@weave.op
def compact_execution_trace(
    trace_tree: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 10000,
) -> str:
    """
    Compact an execution trace using an LLM if it exceeds the token threshold.
    
    Args:
        trace_tree: The formatted trace tree string
        model: LLM model to use for compaction
        max_tokens: Token threshold - if exceeded, compaction is triggered
        
    Returns:
        Original trace if under threshold, or compacted version
    """
    estimated_tokens = estimate_token_count(trace_tree)
    
    if estimated_tokens <= max_tokens:
        return trace_tree
    
    # Calculate target tokens for compaction (aim for 70% of max)
    target_tokens = int(max_tokens * 0.7)
    
    try:
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": TRACE_COMPACTION_SYSTEM_PROMPT},
                {"role": "user", "content": TRACE_COMPACTION_USER_PROMPT.format(
                    target_tokens=target_tokens,
                    trace_tree=trace_tree
                )},
            ],
            temperature=0.0,
        )
        
        compacted = response.choices[0].message.content
        return compacted if compacted else trace_tree
        
    except Exception as e:
        # If compaction fails, return truncated original
        print(f"[yellow]Warning: Trace compaction failed: {e}. Using truncated trace.[/yellow]")
        # Simple truncation fallback
        max_chars = max_tokens * 4
        if len(trace_tree) > max_chars:
            return trace_tree[:max_chars] + "\n... [truncated]"
        return trace_tree
