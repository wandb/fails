import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import litellm
import simple_parsing
import weave
import yaml
from agents import Agent, Runner, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fails.cli.arrow_selector import simple_arrow_selection
from fails.cli.failure_selector import interactive_failure_column_selection
from fails.prompts import (
    CATEGORIZATION_REVIEW_PROMPT,
    CATEGORIZATION_REVIEW_SYSTEM_PROMPT,
    CLUSTERING_PROMPT,
    CLUSTERING_SYSTEM_PROMPT,
    FINAL_CLASSIFICATION_PROMPT,
    FINAL_CLASSIFICATION_SYSTEM_PROMPT,
    FIRST_PASS_CATEGORIZATION_PROMPT,
    FIRST_PASS_CATEGORIZATION_SYSTEM_PROMPT,
    Category,
    CategoryReview,
    ClusteringCategories,
    FinalClassification,
    FinalClassificationResult,
    FirstPassCategorization,
    FirstPassCategorizationResult,
)
from fails.utis import filter_trace_data_by_columns
from fails.weave_query import (
    TraceDepth,
    get_available_columns,
    query_evaluation_data,
)

load_dotenv()
set_tracing_disabled(True)

logging.getLogger("LiteLLM").setLevel(logging.ERROR)
litellm.turn_off_message_logging = True

api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    os.environ["LLM_API_KEY"] = api_key
else:
    raise ValueError("GOOGLE_API_KEY environment variable not set")


@dataclass
class Args:
    """Script arguments for the pipeline."""

    model: str = "gemini/gemini-2.5-pro"
    debug: bool = False
    force_column_selection: bool = False
    config_file: str = os.path.expanduser("~/.wandb/failure_categorization_config.yaml")
    wandb_entity: str = "wandb-applied-ai-team"
    wandb_project: str = "eval-failures"
    wandb_logging_entity: str = "wandb-applied-ai-team"
    wandb_logging_project: str = "eval-failures-testing"


@weave.op
def interactive_column_selection(
    console: Console, columns: list[str], preselected: set[str]
) -> set[str]:
    """
    Interactive column selection using arrow keys.

    Args:
        console: Rich console instance
        columns: List of available columns
        preselected: Set of pre-selected columns

    Returns:
        Set of selected columns
    """
    # Add parent directory to path to find the selector module

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    return simple_arrow_selection(console, columns, preselected)

@weave.op
def load_column_preferences(
    config_file: str, wandb_entity: str, wandb_project: str
) -> Optional[List[str]]:
    """
    Load column preferences for a specific project from config file.

    Args:
        config_file: Path to the config file
        wandb_entity: Weave entity name
        wandb_project: Weave project name

    Returns:
        List of column names if preferences exist, None otherwise
    """
    config_path = Path(config_file)

    if not config_path.exists():
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Look for project-specific config
        project_key = f"{wandb_entity}/{wandb_project}"

        if (
            config
            and project_key in config
            and "failure_categorization_columns" in config[project_key]
        ):
            return config[project_key]["failure_categorization_columns"]

        return None

    except (yaml.YAMLError, KeyError) as e:
        print(f"[yellow]Warning: Error reading config file: {e}[/yellow]")
        return None

@weave.op
def save_column_preferences(
    config_file: str, wandb_entity: str, wandb_project: str, columns: List[str]
) -> None:
    """
    Save column preferences for a specific project to config file.

    Args:
        config_file: Path to the config file
        wandb_entity: Weave entity name
        wandb_project: Weave project name
        columns: List of column names to save
    """
    config_path = Path(config_file)

    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing config or create new
    config = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except yaml.YAMLError:
            print(
                "[yellow]Warning: Existing config file is invalid, creating new one[/yellow]"
            )
            config = {}

    # Update with new preferences
    project_key = f"{wandb_entity}/{wandb_project}"
    if project_key not in config:
        config[project_key] = {}

    config[project_key]["failure_categorization_columns"] = columns

    # Save config as YAML
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"[green]✓ Saved column preferences to {config_file}[/green]")

@weave.op
def load_failure_column_preferences(
    config_file: str, wandb_entity: str, wandb_project: str
) -> Optional[Dict[str, Any]]:
    """
    Load failure column preferences for a specific project from config file.

    Args:
        config_file: Path to the config file
        wandb_entity: Weave entity name
        wandb_project: Weave project name

    Returns:
        Dict with 'failure_column' and 'failure_value' if preferences exist, None otherwise
    """
    config_path = Path(config_file)

    if not config_path.exists():
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Look for project-specific config
        project_key = f"{wandb_entity}/{wandb_project}"

        if (
            config
            and project_key in config
            and "failure_column" in config[project_key]
            and "failure_value" in config[project_key]
        ):
            return {
                "failure_column": config[project_key]["failure_column"],
                "failure_value": config[project_key]["failure_value"]
            }

        return None

    except (yaml.YAMLError, KeyError) as e:
        print(f"[yellow]Warning: Error reading config file: {e}[/yellow]")
        return None

@weave.op
def save_failure_column_preferences(
    config_file: str, 
    wandb_entity: str, 
    wandb_project: str, 
    failure_column: str,
    failure_value: bool
) -> None:
    """
    Save failure column preferences for a specific project to config file.

    Args:
        config_file: Path to the config file
        wandb_entity: Weave entity name
        wandb_project: Weave project name
        failure_column: The column name to filter by
        failure_value: The boolean value to filter for
    """
    config_path = Path(config_file)

    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing config or create new
    config = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except yaml.YAMLError:
            print(
                "[yellow]Warning: Existing config file is invalid, creating new one[/yellow]"
            )
            config = {}

    # Update with new preferences
    project_key = f"{wandb_entity}/{wandb_project}"
    if project_key not in config:
        config[project_key] = {}

    config[project_key]["failure_column"] = failure_column
    config[project_key]["failure_value"] = failure_value

    # Save config as YAML
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"[green]✓ Saved failure column preferences to {config_file}[/green]")

@weave.op
def get_column_preferences(
    config_file: str,
    wandb_entity: str,
    wandb_project: str,
    eval_id: str,
    console: Console,
    force_column_selection: bool = False,
    debug: bool = False,
) -> Tuple[Optional[Dict[str, Any]], Optional[List[str]]]:
    """
    Get failure column and regular column preferences for a specific project.

    Args:
        config_file: Path to the config file
        wandb_entity: Weave entity name
        wandb_project: Weave project name
        eval_id: Evaluation ID
        console: Rich console instance
        force_column_selection: Force re-selection of columns
        debug: Debug mode

    Returns:
        Tuple of (failure_config, columns_list) where:
        - failure_config: Dict with 'failure_column' and 'failure_value' or None
        - columns_list: List of column names for data extraction
    """
    # First, handle failure column selection
    console.print("\n[bold blue]🔍 Failure Column Configuration[/bold blue]")
    
    # Get all available columns first
    column_info = get_available_columns(
        eval_id=eval_id,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        include_nested_paths=True,
        max_nesting_depth=4,
    )
    all_columns = column_info["all_columns"]
    
    # Check for saved failure column preferences
    saved_failure_config = load_failure_column_preferences(config_file, wandb_entity, wandb_project)
    failure_config = None
    
    if saved_failure_config and not force_column_selection:
        # Use saved failure column preferences
        console.print(
            Panel(
                f"[green]Using saved failure column configuration[/green]\n\n"
                f"Column: [cyan]{saved_failure_config['failure_column']}[/cyan]\n"
                f"Filter for: [{'red' if not saved_failure_config['failure_value'] else 'green'}]"
                f"{saved_failure_config['failure_value']}[/{'red' if not saved_failure_config['failure_value'] else 'green'}] values\n\n"
                "[dim]To re-select, re-run with --force-column-selection[/dim]",
                title="🚨 Failure Filter Configuration",
                border_style="yellow",
                padding=(1, 2),
            )
        )
        failure_config = saved_failure_config
    else:
        # Perform failure column selection
        if saved_failure_config and force_column_selection:
            console.print("[yellow]Force selection enabled - overriding saved failure column[/yellow]")
        else:
            console.print("[yellow]No saved failure column preferences found[/yellow]")
        
        # Interactive failure column selection
        failure_column, failure_value = interactive_failure_column_selection(console, all_columns)
        
        if failure_column and failure_value is not None:
            # Validate that the selected column exists and is boolean
            # We'll check this when we query the data
            failure_config = {
                "failure_column": failure_column,
                "failure_value": failure_value
            }
            
            # Save the failure column preferences
            save_failure_column_preferences(
                config_file, wandb_entity, wandb_project, 
                failure_column, failure_value
            )
        else:
            console.print("[yellow]No failure column selected - will process all traces[/yellow]")
    
    # Now handle regular column selection
    saved_columns = load_column_preferences(config_file, wandb_entity, wandb_project)

    if saved_columns and not force_column_selection:
        # Use saved preferences
        console.print("\n[bold blue]📋 Using saved column preferences[/bold blue]")
        selected_columns = set(saved_columns)

        # Show columns grouped
        grouped = {}
        other_cols = []
        for col in sorted(selected_columns):
            if "." in col:
                prefix = col.split(".")[0]
                if prefix not in grouped:
                    grouped[prefix] = []
                grouped[prefix].append(col)
            else:
                other_cols.append(col)

        # Build the column display content
        column_content = f"[green]Using {len(selected_columns)} saved columns for {wandb_entity}/{wandb_project}[/green]\n\n"
        column_content += "[dim]To re-select columns, re-run the script with --force-column-selection[/dim]\n"
        
        # Add grouped columns to content
        for prefix in sorted(grouped.keys()):
            cols = grouped[prefix]
            column_content += f"\n[yellow]{prefix}[/yellow] ({len(cols)} columns):\n"
            for col in cols[:3]:
                column_content += f"  • {col}\n"
            if len(cols) > 3:
                column_content += f"  ... and {len(cols) - 3} more\n"

        if other_cols:
            column_content += f"\n[yellow]Top-level[/yellow] ({len(other_cols)} columns):\n"
            for col in other_cols:
                column_content += f"  • {col}\n"

        # Display everything in one panel
        console.print(
            Panel(
                column_content.rstrip(),
                title="📊 Column Configuration",
                border_style="yellow",
                padding=(1, 2),
            )
        )

    else:
        # Perform column selection
        console.print("\n[bold blue]📋 Column Selection[/bold blue]")

        if saved_columns and force_column_selection:
            console.print(
                "[yellow]Force selection enabled - overriding saved preferences[/yellow]"
            )
        elif not saved_columns:
            console.print(
                "[yellow]No saved preferences found for this project[/yellow]"
            )

        console.print("Using discovered columns for selection...")

        # Filter columns based on user requirements
        # Define metadata columns to exclude (not relevant for analysis)
        metadata_columns = {
            "deleted_at",
            "display_name",
            "storage_size_bytes",
            "thread_id",
            "total_storage_size_bytes",
            "trace_id",
            "turn_id",
            "wb_run_id",
            "wb_run_step",
            "wb_user_id",
            "project_id",
        }

        # Define top-level columns to exclude (we'll show their nested properties instead)
        exclude_top_level = {
            "inputs",
            "output",
        }  # Don't show these as standalone columns

        # Filter out irrelevant columns
        filtered_columns = [
            col
            for col in all_columns
            if not col.startswith("attributes.")
            and not col.startswith("summary.")
            and col not in metadata_columns
            and col not in exclude_top_level  # Exclude top-level objects
            and not any(part.startswith('_') for part in col.split('.'))  # Exclude underscore-prefixed keys
        ]

        # For "other" group, only keep specific columns
        # This will be applied after grouping in the selector
        allowed_other_columns = {"id", "exception", "started_at", "ended_at"}

        # Apply additional filtering for non-dotted columns
        final_filtered_columns = []
        for col in filtered_columns:
            if "." in col:
                # Keep all dotted columns (nested properties)
                final_filtered_columns.append(col)
            else:
                # For non-dotted columns, only keep allowed ones
                if col in allowed_other_columns:
                    final_filtered_columns.append(col)

        filtered_columns = final_filtered_columns

        # Pre-select output.scores.* columns and exception
        preselected = set()
        for col in filtered_columns:
            if col.startswith("output.scores."):
                preselected.add(col)
            elif col == "exception":
                preselected.add(col)

        # Interactive column selection
        selected_columns = interactive_column_selection(
            console, filtered_columns, preselected
        )

        console.print(f"\n[green]✅ Selected {len(selected_columns)} columns[/green]")

        # Save preferences
        save_column_preferences(
            config_file, wandb_entity, wandb_project, list(selected_columns)
        )

        if debug:
            console.print("[dim]Selected columns:[/dim]")
            for col in sorted(selected_columns):
                console.print(f"  - {col}")

    # Always include display_name for better user experience
    columns_for_query = list(selected_columns)
    if "display_name" not in columns_for_query:
        columns_for_query.append("display_name")
    
    # Always include the failure column if one was selected
    if failure_config and failure_config["failure_column"] not in columns_for_query:
        columns_for_query.append(failure_config["failure_column"])
        
    return failure_config, columns_for_query


def construct_first_pass_categorization_prompt(
    row_input: dict | str,
    row_output: dict | str,
    evaluation_evaluation_or_scorer_data: dict | str,
    user_context: str,
) -> str:
    # Convert to JSON strings if needed
    if isinstance(row_input, dict):
        row_input = json.dumps(row_input, indent=2)
    if isinstance(row_output, dict):
        row_output = json.dumps(row_output, indent=2)
    if isinstance(evaluation_evaluation_or_scorer_data, dict):
        evaluation_evaluation_or_scorer_data = json.dumps(evaluation_evaluation_or_scorer_data, indent=2)
        
    first_pass_categorization_prompt_str = FIRST_PASS_CATEGORIZATION_PROMPT.format(
        user_context=user_context,
        row_input=row_input,
        row_output=row_output,
        evaluation_evaluation_or_scorer_data=evaluation_evaluation_or_scorer_data,
    )
    return first_pass_categorization_prompt_str

@weave.op
async def draft_categorization(
    trace_id: str,
    row_input: str | dict,
    row_output: str | dict,
    evaluation_evaluation_or_scorer_data: str | dict,
    user_context: str,
    model: str,
    debug: bool = False,
) -> FirstPassCategorizationResult:
    draft_categorization_llm = Agent(
        name="Row by Row",
        instructions=FIRST_PASS_CATEGORIZATION_SYSTEM_PROMPT,
        model=LitellmModel(model=model, api_key=os.environ["LLM_API_KEY"]),
        output_type=FirstPassCategorization,
    )
    
    first_pass_categorization_prompt_str = construct_first_pass_categorization_prompt(
        user_context=user_context,
        row_input=row_input,
        row_output=row_output,
        evaluation_evaluation_or_scorer_data=evaluation_evaluation_or_scorer_data,
    )

    draft_categorizations = await Runner.run(
        draft_categorization_llm,
        first_pass_categorization_prompt_str,
    )

    draft_categorization_result = FirstPassCategorizationResult(
        trace_id=trace_id,
        thinking=draft_categorizations.final_output.thinking,
        first_pass_categories=draft_categorizations.final_output.first_pass_categories,
    )

    return draft_categorization_result

@weave.op
async def run_draft_categorization(
    trace_data: dict,
    user_context: str,
    model: str,
    debug: bool = False,
) -> list[FirstPassCategorizationResult]:

    # Create tasks with trace_id
    tasks = [
        draft_categorization(
            trace_id=trace_entry["id"],
            row_input=trace_entry["inputs"],
            row_output=trace_entry["output"],
            evaluation_evaluation_or_scorer_data=trace_entry["scores"],
            user_context=user_context,
            model=model,
            debug=debug,
        )
        for trace_entry in trace_data
    ]

    draft_categorization_results = await asyncio.gather(*tasks)

    return draft_categorization_results

def construct_final_classification_prompt(
    row_input: str | dict,
    row_output: str | dict,
    evaluation_evaluation_or_scorer_data: str | dict,
    user_context: str,
    available_categories_str: str,
) -> str:
    
    # Convert dictionaries to JSON strings if needed
    if isinstance(row_input, dict):
        row_input = json.dumps(row_input, indent=2)
    if isinstance(row_output, dict):
        row_output = json.dumps(row_output, indent=2)
    if isinstance(evaluation_evaluation_or_scorer_data, dict):
        evaluation_evaluation_or_scorer_data = json.dumps(evaluation_evaluation_or_scorer_data, indent=2)

    final_classification_prompt_str = FINAL_CLASSIFICATION_PROMPT.format(
        user_context=user_context,
        row_input=row_input,
        row_output=row_output,
        evaluation_evaluation_or_scorer_data=evaluation_evaluation_or_scorer_data,
        available_failure_categories=available_categories_str,
    )
    return final_classification_prompt_str

@weave.op
async def final_classification(
    trace_id: str,
    row_input: str,
    row_output: str,
    evaluation_evaluation_or_scorer_data: str,
    user_context: str,
    available_categories_str: str,
    model: str,
) -> FinalClassificationResult:
    
    final_classification_llm = Agent(
        name="Final Classification",
        instructions=FINAL_CLASSIFICATION_SYSTEM_PROMPT,
        model=LitellmModel(model=model, api_key=os.environ["LLM_API_KEY"]),
        output_type=FinalClassification,
    )

    final_classification_prompt_str = construct_final_classification_prompt(
        row_input=row_input,
        row_output=row_output,
        evaluation_evaluation_or_scorer_data=evaluation_evaluation_or_scorer_data,
        user_context=user_context,
        available_categories_str=available_categories_str,
    )

    classification_result = await Runner.run(
        final_classification_llm, final_classification_prompt_str
    )

    final_classification_result = FinalClassificationResult(
        trace_id=trace_id,
        thinking=classification_result.final_output.thinking,
        selected_category=classification_result.final_output.selected_category,
        confidence_score=classification_result.final_output.confidence_score,
        classification_notes=classification_result.final_output.classification_notes,
    )

    return final_classification_result

@weave.op
async def run_final_classification(
    trace_data: list[dict],
    user_context: str,
    available_categories_str: str,
    model: str,
    debug: bool = False,
) -> list[FinalClassificationResult]:
    
    classification_tasks = [
        final_classification(
            trace_id=trace_entry["id"],
            row_input=trace_entry["inputs"],
            row_output=trace_entry["output"],
            evaluation_evaluation_or_scorer_data=trace_entry["scores"],
            user_context=user_context,
            available_categories_str=available_categories_str,
            model=model,
        )
        for trace_entry in trace_data
    ]

    final_classification_results = await asyncio.gather(*classification_tasks)

    return final_classification_results


def construct_clustering_prompt(
    draft_categorizations_and_notes: str,
    num_traces: int,
) -> str:
    clustering_prompt_str = CLUSTERING_PROMPT.format(
        num_traces=num_traces,
        draft_categorizations_and_notes=draft_categorizations_and_notes,
    )
    return clustering_prompt_str

@weave.op
async def aggregate_categorizations(
    draft_categorization_results_str: str,
    num_draft_categorizations: int,
    user_context: str,
    model: str,
    debug: bool = False,
) -> ClusteringCategories:

    clustering_system_prompt_str = CLUSTERING_SYSTEM_PROMPT.format(
        num_traces=num_draft_categorizations
    )

    clustering_prompt_str = construct_clustering_prompt(
        draft_categorizations_and_notes=draft_categorization_results_str,
        num_traces=num_draft_categorizations,
    )

    review_categorizations_llm = Agent(
        name="Review Agent",
        instructions=clustering_system_prompt_str,
        model=LitellmModel(model=model, api_key=os.environ["LLM_API_KEY"]),
        output_type=ClusteringCategories,
    )

    review_result = await Runner.run(
        review_categorizations_llm,
        clustering_prompt_str,
    )

    return review_result.final_output


@weave.op
async def run_pipeline(
    trace_data: list[dict],
    user_context: str,
    model: str,
    debug: bool = False,
    console: Console = Console(),
):
    # ----------------- STEP 1: Draft categorization -----------------
    console.print("[bold blue]📝 STEP 1: Starting draft categorization...[/bold blue]")

    if debug:
        first_pass_categorization_prompt_str = construct_first_pass_categorization_prompt(
            row_input=trace_data[0]["inputs"],
            row_output=trace_data[0]["output"],
            evaluation_evaluation_or_scorer_data=trace_data[0]["scores"],
            user_context=user_context,
        )
        console.print(
            Panel(
                FIRST_PASS_CATEGORIZATION_SYSTEM_PROMPT,
                title="🤖 First Pass Categorization System Prompt",
                border_style="blue",
                padding=(1, 2),
            )
        )
        console.print(
            Panel(
                first_pass_categorization_prompt_str,
                title="💭 First Pass Categorization Prompt",
                border_style="blue",
                padding=(1, 2),
            )
        )

    draft_categorization_results = await run_draft_categorization(
        trace_data=trace_data,
        user_context=user_context,
        model=model,
        debug=debug,
    )

    num_draft_categorizations = len(draft_categorization_results)

    # Create a nice display for draft categorization results
    console.print(
        Panel(
            f"[bold green]✅ Completed {num_draft_categorizations} draft categorizations[/bold green]",
            title="Draft Categorization Results",
            border_style="green",
        )
    )

    # ----------------- STEP 2: Review categorizations -----------------

    console.print("\n[bold blue]🔍 STEP 2: Reviewing categorizations...[/bold blue]")

    # Create resclustering prompt (needed for the agent)
    draft_categorization_results_str = "\n" + "=" * 80 + "\n"

    for c_i, draft_categorization_result in enumerate(draft_categorization_results):
        draft_categorization_results_str += (
            f"### Evaluation Trace ID: {draft_categorization_result.trace_id}\n\n"
        )

        if debug:
            result_table = Table(show_header=True, box=None, padding=(0, 1))
            result_table.add_column("Candidate Category", style="cyan", width=120)
            result_table.add_column("Category Description", style="white", width=120)
            result_table.add_column("Eval Failure Note", style="dim", width=120)

        for i, first_pass_category in enumerate(draft_categorization_result.first_pass_categories):
            draft_categorization_results_str += (
                f"#### Candiate Category Name {i + 1}:\n\n{first_pass_category.category_name}\n\n"
            )
            draft_categorization_results_str += (
                f"#### Category Description {i + 1}: {first_pass_category.category_description}\n\n"
            )
            draft_categorization_results_str += (
                f"#### Eval Failure Note {i + 1}\n\n{first_pass_category.eval_failure_note}\n\n"
            )
            if debug:
                result_table.add_row(first_pass_category.category_name, first_pass_category.category_description, first_pass_category.eval_failure_note)
        
        draft_categorization_results_str += "\n" + "=" * 80 + "\n"

        if debug:
            console.print(
                Panel(
                    result_table, title=f"[green]Trace {c_i + 1}[/green]", border_style="green"
                )
            )

    if debug:
        console.print(
            Panel(
                CLUSTERING_SYSTEM_PROMPT.format(num_traces=num_draft_categorizations),
                title="🤖 Clustering System Prompt",
                border_style="blue",
                padding=(1, 2),
            )
        )
        clustering_prompt_str = construct_clustering_prompt(
            draft_categorizations_and_notes=draft_categorization_results_str,
            num_traces=num_draft_categorizations,
        )
        console.print(
            Panel(
                clustering_prompt_str, 
                title="💭 Clustering Prompt", 
                border_style="blue",
                padding=(1, 2),
            )
        )

    review_data = await aggregate_categorizations(
        draft_categorization_results_str=draft_categorization_results_str,
        num_draft_categorizations=num_draft_categorizations,
        user_context=user_context,
        model=model,
        debug=debug,
    )

                        # Pretty print the review result
    console.print(
        Panel(
            "[bold green]✨ Review completed successfully![/bold green]",
            title="Review Result",
            border_style="green",
        )
    )

    console.print("=" * 80)
    console.print("Candidate categories:")
    console.print("-" * 80)
    console.print(review_data.category_long_list_thinking)
    console.print("-" * 80)
    for category in review_data.task_failure_categories:
        console.print(f"Category: {category.category_name}")
        console.print(f"Description: {category.category_description}")
        console.print(f"Notes: {category.category_notes}")
        console.print("-" * 80)


    # ----------------- STEP 3: Final classification -----------------

    console.print("\n[bold blue]🎯 STEP 3: Final classification of failures...[/bold blue]")

    # Add "other" category to the list
    all_categories = review_data.task_failure_categories + [
        Category(
            thinking="This is the default category for failures that don't fit into any other category",
            category_name="other",
            category_description="Can be used if the evaluation failure sample can't be classified into one of the other classes",
            category_notes="Default category for unclassifiable failures"
        )
    ]

    # Format categories for the prompt
    categories_str = ""
    for i, category in enumerate(all_categories):
        categories_str += f"\n### Category {i + 1}: {category.category_name}\n"
        categories_str += f"**Description:** {category.category_description}\n"
        categories_str += f"**Notes:** {category.category_notes}\n"

    if debug:
        final_classification_prompt_str = construct_final_classification_prompt(
            row_input=trace_data[0]["inputs"],
            row_output=trace_data[0]["output"],
            evaluation_evaluation_or_scorer_data=trace_data[0]["scores"],
            user_context=user_context,
            available_categories_str=categories_str,
        )
        console.print(
            Panel(
                FINAL_CLASSIFICATION_SYSTEM_PROMPT,
                title="🤖 Final Classification System Prompt",
                border_style="blue",
                padding=(1, 2),
            )
        )
        console.print(
            Panel(
                final_classification_prompt_str,
                title="💭 Final Classification Prompt",
                border_style="blue",
                padding=(1, 2),
            )
        )

    final_classification_results = await run_final_classification(
        trace_data=trace_data,
        user_context=user_context,
        available_categories_str=categories_str,
        model=model,
        debug=debug,
    )
    return final_classification_results, all_categories



@weave.op
async def run_extract_and_classify_pipeline(
    eval_id: str,
    user_context: str,
    debug: bool,
    model: str,
    config_file: str,
    wandb_entity: str,
    wandb_project: str,
    force_column_selection: bool = False,
):
    # Query Weave for evaluation data using the enhanced API
    console = Console()

    console.print("[bold cyan]🔍 Fetching evaluation trace...[/bold cyan]")

    if debug:
        console.print(
            f"[bold red]🔍 DEBUG MODE ENABLED, switching to {model}[/bold red]"
        )
        model = "gemini/gemini-2.5-flash"
        N_FIRST_PASS_TRACES = 3
    else:
        N_FIRST_PASS_TRACES = None
    
    # Display user context in a nice yellow box
    console.print(
        Panel(
            user_context,
            title="📝 User Context",
            border_style="yellow",
            padding=(1, 2),
        )
    )

    # ----------------- Column Selection -----------------

    # Check for saved column preferences
    failure_config, columns_for_query = get_column_preferences(
        config_file=config_file,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        eval_id=eval_id,
        debug=debug,
        force_column_selection=force_column_selection,
        console=console,
    )

    eval_data = query_evaluation_data(
        eval_id=eval_id,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        columns=columns_for_query,  # Use the selected columns + display_name
        include_outputs=True,
        deep_ref_extraction=False,
        trace_depth=TraceDepth.DIRECT_CHILDREN,  # Get evaluation + direct children
        include_hierarchy=True,
        limit=N_FIRST_PASS_TRACES,
        filter_dict={failure_config["failure_column"]: failure_config["failure_value"]} if failure_config else None,
    )
    
    # Filter the child traces to only include selected columns
    if "children" in eval_data and eval_data["children"]:
        eval_data["children"] = filter_trace_data_by_columns(
            eval_data["children"], 
            columns_for_query
        )

    op_name = eval_data["evaluation"].get("op_name", "Unknown")
    # Truncate long op names
    if len(op_name) > 100:
        op_name = op_name[:90] + "..."

    # Use a panel for the evaluation summary
    eval_info = f"""[bold green]Evaluation ID:[/bold green] {eval_data["evaluation"]["id"]}
[bold green]Op Name:[/bold green] {op_name}
[bold green]Total traces:[/bold green] {eval_data["trace_count"]["total"]}
[bold green]Direct children:[/bold green] {eval_data["trace_count"].get("direct_children", 0)}"""

    # If we have a failure filter, add info about filtered results
    if failure_config:
        eval_info += f"\n[bold green]Failure filter:[/bold green] {failure_config['failure_column']} == {failure_config['failure_value']}"
        eval_info += f"\n[bold green]Filtered children:[/bold green] {len(eval_data.get('children', []))}"

    console.print(Panel(eval_info, title="📊 Evaluation Summary", border_style="green"))

    # Validate failure column if one was selected
    if failure_config and eval_data.get("children"):
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

    # Show evaluation summary if available
    if "summary" in eval_data["evaluation"]:
        console.print(
            f"[yellow]Evaluation Summary: {eval_data['evaluation']['summary']}[/yellow]"
        )

    # console.print(f"[dim]Evaluation data keys:[/dim] {', '.join(eval_data.keys())}")
    # console.print(f"[dim]Evaluation hierarchy:[/dim] {eval_data['hierarchy']}")

    console.print(
        f"[dim]First child keys:[/dim] {', '.join(eval_data['children'][0].keys())}\n"
    )
    trace_data = []

    console.print("\n[dim]CHILDREN:[/dim]")
    console.print(
        f"[dim]{len(eval_data['children'])} children found, sampling first 3:[/dim]\n"
    )
    for i, trace in enumerate(eval_data["children"][:3]):
        # Since we've already filtered the data, we can use it directly
        trace_entry = {
            "id": trace.get("id"),
            "inputs": trace.get("inputs", {}),
            "output": trace.get("output", {}),
            "scores": trace.get("output", {}).get("scores", {}) if trace.get("output") else {},
        }

        trace_data.append(trace_entry)

        if debug and i == 0:
            # Create a table for trace details
            trace_table = Table(
                title=f"Trace {i + 1} Details",
                show_header=True,
                header_style="bold magenta",
            )
            trace_table.add_column("Property", style="cyan", width=20)
            trace_table.add_column("Value", style="white")

            trace_table.add_row("ID", trace_entry.get("id", "N/A"))
            trace_table.add_row("Name", str(trace_entry.get("display_name", "N/A")))
            trace_table.add_row("Op Name", trace_entry.get("op_name", "N/A"))
            trace_table.add_row("Started At", trace_entry.get("started_at", "N/A"))
            trace_table.add_row("Ended At", trace_entry.get("ended_at", "N/A"))
            trace_table.add_row("Summary", "\n" + str(trace_entry.get("summary", "N/A")))
            # trace_table.add_row("Input", "\n" + str(trace_entry.get("inputs", {})))
            
            if (
                trace_entry.get("output")
                and isinstance(trace_entry.get("output"), dict)
                and "output" in trace_entry["output"]
            ):
                trace_table.add_row("Output", "\n" + str(trace["output"]["output"]))
            console.print(trace_table)
            console.print("[dim]" + "─" * 50 + "[/dim]\n")


    console.print("\n[bold cyan]" + "═" * 50 + "[/bold cyan]\n")

    final_classification_results, all_categories = await run_pipeline(
        trace_data=trace_data,
        user_context=user_context,
        model=model,
        debug=debug,
        console=console,
    )

    # ----------------- Generate Evaluation Report -----------------

    console.print("\n[bold blue]📊 STEP 4: Generating evaluation report...[/bold blue]")
    
    # Create a summary of classifications
    classification_summary = {}
    total_failures = len(final_classification_results)
    
    for result in final_classification_results:
        category = result.selected_category
        if category not in classification_summary:
            classification_summary[category] = {
                "traces": [],
                "category_info": None
            }
        classification_summary[category]["traces"].append({
            "trace_id": result.trace_id,
            "confidence": result.confidence_score,
            "notes": result.classification_notes
        })
    
    # Get category info from all_categories
    for category in all_categories:
        if category.category_name in classification_summary:
            classification_summary[category.category_name]["category_info"] = category
    
    # Sort categories by count (descending)
    sorted_categories = sorted(
        classification_summary.items(), 
        key=lambda x: len(x[1]["traces"]), 
        reverse=True
    )
    
    # Generate report
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Get evaluation name from eval_data
    eval_name = eval_data["evaluation"].get("display_name", eval_id)
    
    report = f"## {eval_name} Evaluation Report - {current_time}\n\n"
    
    # Add summary table to the report
    # Calculate max widths for alignment
    max_category_width = max(len(category_name.replace("_", " ").title()) for category_name in classification_summary.keys())
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
            report += f"{category_info.category_description}\n\n"
        
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
    
    # Display the report
    console.print("\n" + "=" * 80)
    console.print(report)
    console.print("=" * 80)
    
    console.print("\n[bold green]✅ Pipeline completed successfully![/bold green]")



if __name__ == "__main__":
    eval_id = "0197a72d-2704-7ced-8c07-0fa1e0ab0557"

    # User AI System Context
    USER_AI_SYSTEM_CONTEXT = """My app is trying to idenify insights from transcripts of \
meetings between prospects and our sales team."""

    USER_EVAL_CONTEXT = """To classify speaker IDs from a transcript into whether they \
are from our company or are a customer/prospect."""

    user_context_str = f"""
## User AI System Context

What the user is trying to achieve with their AI system: 

<user_ai_system_context>
{USER_AI_SYSTEM_CONTEXT}
</user_ai_system_context>

## User Eval Context 

What the user is trying to evaluate in their AI system: 

<user_eval_context>
{USER_EVAL_CONTEXT}
</user_eval_context>

"""

    model = "gemini/gemini-2.5-pro"

    args: Args = simple_parsing.parse(Args)

    if not user_context_str:
        raise ValueError(
            "User context is required. Please provide a user context about the AI system and what they are trying to evaluate."
        )

    weave.init(f"{args.wandb_logging_entity}/{args.wandb_logging_project}")

    asyncio.run(
        run_extract_and_classify_pipeline(
            eval_id=eval_id,
            user_context=user_context_str,
            debug=args.debug,
            model=model,
            config_file=args.config_file,
            force_column_selection=args.force_column_selection,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
        )
    )
