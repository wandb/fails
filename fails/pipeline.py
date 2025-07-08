import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import litellm
import simple_parsing
import yaml
from agents import Agent, Runner, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel
from cli.arrow_selector import simple_arrow_selection
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fails.weave_query import (
    TraceDepth,
    get_available_columns,
    query_evaluation_data,
)

load_dotenv()
set_tracing_disabled(True)

logging.getLogger("LiteLLM").setLevel(logging.ERROR)
litellm.turn_off_message_logging = True


@dataclass
class Args:
    """Script arguments for the pipeline."""

    model: str = "gemini/gemini-2.5-pro"
    api_key: str | None = os.getenv("GOOGLE_API_KEY")
    debug: bool = False
    force_column_selection: bool = False
    config_file: str = os.path.expanduser("~/.wandb/failure_categorization_config.yaml")


EVALUATION_FAILURE_DEFINITION = """An evaluation failure is defined as the output of a single row \
that failed the evaluator or scorer criteria. An individual row that failed can evaluation might do so for a \
number of reasons such as:

- The output was judged to be incorrect by the evaluator or scorer
- The output was not formatted correctly.
- The output had a code execution error.
- etc.
"""

FIRST_PASS_CATEGORIZATION_SYSTEM_PROMPT = f"""
# Task - Evaluation Failure Categorization

Your task is to output a draft set of notes and candiate task failure categories given evaluation failures \
data from a users AI system. We are trying to help a user understand the nature of the failures in their AI system \
and identify the root causes of the failures.

## Evaluation Failure Definition

{EVALUATION_FAILURE_DEFINITION}

### How your notes and candidate task failure categories will be used

With this rough draft of failure categories and notes for 1 or a small number of rows, a later step in this pipeline \
will subsequently compare the draft notes and candidate task failure categories across a larger number of rows. \
From here, we will iteratively align and refine the notes and candidate task failure categories until we \
have a set of notes and candidate task failure categories that are consistent across a larger number of rows.

## Inspiration - Open Coding

This task is similar to open coding, where we are trying to identify the underlying issue and phenomenon:

> Open coding attempts to codify, name or classifying the observed phenomenon and is achieved by segmenting \
data into meaningful expressions and describing that data with a single word or short sequence of words

Some examples of open coding questions to consider when drafting the notes and candidate task failure categories:

- Identify the underlying issue and phenomenon *(What?*)
- Identify the phenomenon's attributes *(What kind?*)
- Determine the time, course and location of the failure *(When? How long? Where?)*
- Identify the intensity of the failure (*How much? How long?*)
- Identify the reasons attached to the failure (*Why?*)
- Identify intention or purpose of the failure (*Why?)*

Take inspiration from the above open coding questions but there is no need to be exhaustive if its not relevant \
to the failure data in question.
"""

FIRST_PASS_CATEGORIZATION_PROMPT = """
Given the specific task context from the user as well as the evaluation failure data, please make your best \
guess at the notes and candidate task failure categories for the given row input and row output.

## User context about their AI system

Below is the context from the user about their AI system and what they are trying to evaluate. This might add \
context to the evaluation failure data below and help you better understand what the user is trying to achieve with \
their AI system.

<user_context>
{user_context}
</user_context>

## Evaluation Failure Data

### Inputs that were given to the system
<row_input>
{{row_input}}
</row_input>

### Outputs that were evaluated to be failures
<row_output>
{{row_output}}
</row_output>

### Evaluation or Scorer data and metadata

<evaluation_evaluation_or_scorer_data>
{{evaluation_evaluation_or_scorer_data}}
</evaluation_evaluation_or_scorer_data>

## Analyse

With the above user context and evaluation failure data, please output a draft set of notes and candidate \
task failure categories for the given row input and row output.
"""


class FirstPassCategorization(BaseModel):
    """First pass classification of a single evaluation failure."""

    thinking: str = Field(
        description="A detailed thinking process of the classification."
    )
    notes: str = Field(description="A sentence or two of notes for the classification.")
    candidate_task_failure_categories: list[str] = Field(
        description="""A list of candidate task failure categories.\
Keep all category names lowercase, concise and separated by '_'. If a trace doesn't fit into any of the defined task \
failure categories, it should be classified as "other"."""
    )


class FirstPassCategorizationResult(FirstPassCategorization):
    """First pass classification of a single evaluation failure."""

    trace_id: str = Field(description="The ID of the trace that was classified.")


# ----------------- Clustering draft categorizations -----------------

MAX_N_TASK_FAILURE_CATEGORIES = 7

CLUSTERING_SYSTEM_PROMPT = f"""# Task - Clustering Draft Categorizations

Given {{num_traces}} of draft categorizations and notes for a set of evaluation failures, cluster \
the categorizations and notes into a defined set of task failure categories.

## Definition - Evaluation Failure

{EVALUATION_FAILURE_DEFINITION}

## Task Context - Clustering Draft Categorizations

The purpose of this task is examine draft categorizations and notes for a set of evaluation failures and cluster the \
categories into a canonical set of task failure categories. The aim is to find a set of task failure categories that \
are consistent across a large number of evaluation failures, ideally we have no more than \
{MAX_N_TASK_FAILURE_CATEGORIES} eval failure categories.

If a trace doesn't fit into any of the defined task failure categories, it should be classified as "other".

Keep all category names lowercase, concise and separated by '_'.
"""

CLUSTERING_PROMPT = f"""
## Draft Categorizations and Notes

Here are the draft categorizations and notes for {{num_traces}} traces:

<draft_categorizations_and_notes>

{{draft_categorizations_and_notes}}
</draft_categorizations_and_notes>

## Output

Output a list of maximum {MAX_N_TASK_FAILURE_CATEGORIES} task failure categories - you can output less than \
{MAX_N_TASK_FAILURE_CATEGORIES} if you think that's appropriate.
"""


class Category(BaseModel):
    """A task failure category."""

    thinking: str = Field(
        description="A detailed reasoning process behind the selection of the category \
name, description and notes."
    )
    category_name: str = Field(
        description="""The name of the task failure category. Keep all category \
names lowercase, concise and separated by '_'. If a trace doesn't fit into any of the defined task failure \
categories, it should be classified as 'other'."""
    )
    category_description: str = Field(
        description="A short description of the task failure category."
    )
    category_notes: str = Field(
        description="A sentence or two of notes for the task failure category."
    )


class ClusteringCategories(BaseModel):
    """Clustering of draft categorizations and notes into a set of task failure categories."""

    category_long_list_thinking: str = Field(
        description="A detailed reasoning process and final decision making \
for the selection of the task failure categories."
    )
    task_failure_categories: list[Category] = Field(
        description="""A list of task failure categories. \
If a trace doesn't fit into any of the defined task failure categories, it should be classified as "other"."""
    )


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


def load_column_preferences(
    config_file: str, entity_name: str, project_name: str
) -> Optional[List[str]]:
    """
    Load column preferences for a specific project from config file.

    Args:
        config_file: Path to the config file
        entity_name: Weave entity name
        project_name: Weave project name

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
        project_key = f"{entity_name}/{project_name}"

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


def save_column_preferences(
    config_file: str, entity_name: str, project_name: str, columns: List[str]
) -> None:
    """
    Save column preferences for a specific project to config file.

    Args:
        config_file: Path to the config file
        entity_name: Weave entity name
        project_name: Weave project name
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
    project_key = f"{entity_name}/{project_name}"
    if project_key not in config:
        config[project_key] = {}

    config[project_key]["failure_categorization_columns"] = columns

    # Save config as YAML
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"[green]✓ Saved column preferences to {config_file}[/green]")


async def run_pipeline(
    eval_id: str,
    user_context: str,
    debug: bool,
    model: str,
    api_key: str | None,
    config_file: str,
    force_column_selection: bool = False,
):
    # Query Weave for evaluation data using the enhanced API
    console = Console()

    console.print("[bold cyan]🔍 Fetching evaluation trace...[/bold cyan]")

    # ----------------- Column Selection -----------------

    entity_name = "wandb-applied-ai-team"
    project_name = "eval-failures"

    # Check for saved column preferences
    saved_columns = load_column_preferences(config_file, entity_name, project_name)

    if saved_columns and not force_column_selection:
        # Use saved preferences
        console.print("\n[bold blue]📋 Using saved column preferences[/bold blue]")
        selected_columns = set(saved_columns)

        # Display the columns being used
        console.print(
            Panel(
                f"[green]Using {len(selected_columns)} saved columns for {entity_name}/{project_name}[/green]\n\n"
                "[dim]To re-select columns, use --force-column-selection[/dim]",
                title="📊 Column Configuration",
                border_style="blue",
            )
        )

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

        # Display grouped columns
        for prefix in sorted(grouped.keys()):
            cols = grouped[prefix]
            console.print(f"\n[yellow]{prefix}[/yellow] ({len(cols)} columns):")
            for col in cols[:3]:
                console.print(f"  • {col}")
            if len(cols) > 3:
                console.print(f"  ... and {len(cols) - 3} more")

        if other_cols:
            console.print(f"\n[yellow]Top-level[/yellow] ({len(other_cols)} columns):")
            for col in other_cols:
                console.print(f"  • {col}")

        console.print()

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

        console.print("Discovering available columns...")

        # Get available columns (with nested paths like inputs.field1, inputs.field2, etc.)
        column_info = get_available_columns(
            eval_id=eval_id,
            entity_name=entity_name,
            project_name=project_name,
            include_nested_paths=True,  # This expands objects to show their properties
            max_nesting_depth=4,  # Allow deeper nesting to see paths like inputs.self.transcript
        )

        # Filter columns based on user requirements
        all_columns = column_info["all_columns"]

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
            config_file, entity_name, project_name, list(selected_columns)
        )

        if debug:
            console.print("[dim]Selected columns:[/dim]")
            for col in sorted(selected_columns):
                console.print(f"  - {col}")

    if debug:
        console.print(
            f"[bold red]🔍 DEBUG MODE ENABLED, switching to {model}[/bold red]"
        )
        model = "gemini/gemini-2.5-flash"
        console.print(Panel(user_context, title="User Context", border_style="yellow"))
        first_pass_trace_limit = 3
    else:
        first_pass_trace_limit = None

    # Always include display_name for better user experience
    columns_for_query = list(selected_columns)
    if "display_name" not in columns_for_query:
        columns_for_query.append("display_name")

    eval_data = query_evaluation_data(
        eval_id=eval_id,
        entity_name=entity_name,
        project_name=project_name,
        columns=columns_for_query,  # Use the selected columns + display_name
        include_outputs=True,
        deep_ref_extraction=False,
        trace_depth=TraceDepth.DIRECT_CHILDREN,  # Get evaluation + direct children
        include_hierarchy=True,
        limit=first_pass_trace_limit,
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

    console.print(Panel(eval_info, title="📊 Evaluation Summary", border_style="green"))

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
        f"[dim]{len(eval_data['children'])} children found, sampling first 3:[/dim]"
    )
    for i, trace in enumerate(eval_data["children"][:3]):
        if debug:
            # Create a table for trace details
            trace_table = Table(
                title=f"Trace {i + 1} Details",
                show_header=True,
                header_style="bold magenta",
            )
            trace_table.add_column("Property", style="cyan", width=20)
            trace_table.add_column("Value", style="white")

            trace_table.add_row("ID", trace.get("id", "N/A"))
            trace_table.add_row("Name", str(trace.get("display_name", "N/A")))
            trace_table.add_row("Op Name", trace.get("op_name", "N/A"))
            trace_table.add_row("Started At", trace.get("started_at", "N/A"))
            trace_table.add_row("Ended At", trace.get("ended_at", "N/A"))
            trace_table.add_row("Summary", "\n" + str(trace.get("summary", "N/A")))
            trace_table.add_row("Input", "\n" + str(trace.get("inputs", {})))
            if (
                trace.get("output")
                and isinstance(trace.get("output"), dict)
                and "output" in trace["output"]
            ):
                trace_table.add_row("Output", "\n" + str(trace["output"]["output"]))

            console.print(trace_table)

            console.print("[dim]" + "─" * 50 + "[/dim]\n")

        # Extract data based on available columns
        trace_entry = {
            "id": trace.get("id"),
            "input": trace.get("inputs", {}),
            "output": trace.get("output", {}),
            "scores": {},
        }

        # Safely extract nested fields
        if isinstance(trace.get("output"), dict):
            output_data = trace["output"]

            # Extract scores if available
            if "scores" in output_data:
                trace_entry["scores"] = output_data["scores"]

            # Extract affiliation_score if available
            if "affiliation_score" in output_data:
                trace_entry["output"]["affiliation"] = output_data["affiliation_score"]

            # Extract nested output fields if available
            if "output" in output_data and isinstance(output_data["output"], dict):
                if "reasoning" in output_data["output"]:
                    trace_entry["output"]["reasoning"] = output_data["output"][
                        "reasoning"
                    ]
                if "affiliation" in output_data["output"]:
                    trace_entry["output"]["affiliation"] = output_data["output"][
                        "affiliation"
                    ]

        trace_data.append(trace_entry)

    console.print("\n[bold cyan]" + "═" * 50 + "[/bold cyan]\n")

    # ----------------- STEP 1: Draft categorization -----------------

    console.print("[bold blue]📝 STEP 1: Starting draft categorization...[/bold blue]")

    async def draft_categorization(
        trace_id: str,
        row_input: str,
        row_output: str,
        evaluation_evaluation_or_scorer_data: str,
        user_context: str,
        api_key: str | None,
        model: str,
        debug: bool = False,
    ) -> FirstPassCategorizationResult:
        draft_categorization_llm = Agent(
            name="Row by Row",
            instructions=FIRST_PASS_CATEGORIZATION_SYSTEM_PROMPT,
            model=LitellmModel(model=model, api_key=api_key),
            output_type=FirstPassCategorization,
        )
        first_pass_categorization_prompt_str = FIRST_PASS_CATEGORIZATION_PROMPT.format(
            user_context=user_context,
            row_input=row_input,
            row_output=row_output,
            evaluation_evaluation_or_scorer_data=evaluation_evaluation_or_scorer_data,
        )
        if debug:
            console.print(
                Panel(
                    FIRST_PASS_CATEGORIZATION_SYSTEM_PROMPT,
                    title="First Pass Categorization System Prompt",
                    border_style="blue",
                )
            )
            console.print(
                Panel(
                    first_pass_categorization_prompt_str,
                    title="First Pass Categorization Prompt",
                    border_style="blue",
                )
            )

        draft_categorization_result = await Runner.run(
            draft_categorization_llm,
            first_pass_categorization_prompt_str,
        )
        draft_categorization_result = FirstPassCategorizationResult(
            trace_id=trace_id,
            thinking=draft_categorization_result.final_output.thinking,
            notes=draft_categorization_result.final_output.notes,
            candidate_task_failure_categories=draft_categorization_result.final_output.candidate_task_failure_categories,
        )
        return draft_categorization_result

    # Create tasks with trace_id
    tasks = [
        draft_categorization(
            trace_id=trace_entry["id"],
            row_input=trace_entry["input"],
            row_output=trace_entry["output"],
            evaluation_evaluation_or_scorer_data=trace_entry["scores"],
            user_context=user_context,
            api_key=api_key,
            model=model,
            debug=debug,
        )
        for trace_entry in trace_data
    ]

    console.print("[yellow]⚡ Running categorization tasks...[/yellow]")
    draft_categorization_results = await asyncio.gather(*tasks)
    num_draft_categorizations = len(draft_categorization_results)

    # Create string for clustering prompt (needed for the agent)
    draft_categorization_results_str = ""

    for draft_categorization_result in draft_categorization_results:
        draft_categorization_results_str += (
            f"### Trace ID: {draft_categorization_result.trace_id}\n\n"
        )
        draft_categorization_results_str += (
            f"#### Notes\n\n{draft_categorization_result.notes}\n\n"
        )
        draft_categorization_results_str += f"#### Candidate Task Failure Categories\n\n{draft_categorization_result.candidate_task_failure_categories}\n"
        draft_categorization_results_str += "\n" + "=" * 50 + "\n"

    # Create a nice display for draft categorization results
    console.print(
        Panel(
            f"[bold green]✅ Completed {num_draft_categorizations} draft categorizations[/bold green]",
            title="Draft Categorization Results",
            border_style="green",
        )
    )

    for i, draft_categorization_result in enumerate(draft_categorization_results):
        result_table = Table(show_header=False, box=None, padding=(0, 1))
        result_table.add_column("", style="bold cyan", width=25)
        result_table.add_column("", style="white")

        result_table.add_row("Trace ID", draft_categorization_result.trace_id)
        result_table.add_row("Notes", draft_categorization_result.notes)
        result_table.add_row(
            "Categories",
            ", ".join(draft_categorization_result.candidate_task_failure_categories),
        )

        console.print(
            Panel(
                result_table, title=f"[bold]Result {i + 1}[/bold]", border_style="blue"
            )
        )

    # ----------------- STEP 2: Review categorizations -----------------

    console.print("\n[bold blue]🔍 STEP 2: Reviewing categorizations...[/bold blue]")

    clustering_system_prompt_str = CLUSTERING_SYSTEM_PROMPT.format(
        num_traces=num_draft_categorizations
    )

    clustering_prompt_str = CLUSTERING_PROMPT.format(
        num_traces=num_draft_categorizations,
        draft_categorizations_and_notes=draft_categorization_results_str,
    )

    review_categorizations_llm = Agent(
        name="Review Agent",
        instructions=clustering_system_prompt_str,
        model=LitellmModel(model=model, api_key=api_key),
        output_type=ClusteringCategories,
    )

    if debug:
        console.print(
            Panel(
                clustering_system_prompt_str,
                title="Clustering System Prompt",
                border_style="blue",
            )
        )
        console.print(
            Panel(clustering_prompt_str, title="Clustering Prompt", border_style="blue")
        )

    review_result = await Runner.run(
        review_categorizations_llm,
        clustering_prompt_str,
    )

    # Pretty print the review result
    console.print(
        Panel(
            "[bold green]✨ Review completed successfully![/bold green]",
            title="Review Result",
            border_style="green",
        )
    )

    review_data = review_result.final_output.model_dump()

    # Display task failure categories in a nice table
    if "task_failure_categories" in review_data:
        categories_table = Table(
            title="Clustered Task Failure Categories",
            show_header=True,
            header_style="bold magenta",
        )
        categories_table.add_column("Category", style="cyan", width=20)
        categories_table.add_column("Description", style="white", width=40)
        categories_table.add_column("Notes", style="dim", width=40)

        for category in review_data["task_failure_categories"]:
            categories_table.add_row(
                category["category_name"],
                category["category_description"],
                category["category_notes"],
            )

        console.print(categories_table)

    # ----------------- STEP 3: Final categorization -----------------

    # Run a final test categorization
    console.print("\n[bold blue]🧪 Running final test categorization...[/bold blue]")

    CATEGORIZATION_REVIEW_SYSTEM_PROMPT = """
You are a helpful assistant that categorizes task failure categories.

Given a proposed list of evaluation failure categories and the eval failtures themselves, determine \
if the proposed categories are appropriate.

If the proposed categories are appropriate, return the proposed categories.

If the proposed categories are not appropriate, return a new list of categories that are appropriate \
as well as a note explaining why the proposed categories are not appropriate.
"""

    CATEGORIZATION_REVIEW_PROMPT = """
Given the following user context and evaluation failure data, please output a draft set of notes and candidate \
task failure categories for the given row input and row output.

## User Context

<user_context>
{user_context}
</user_context>

## Evaluation Failure Data

### Inputs that were given to the system
<row_input>
{{row_input}}
</row_input>

### Outputs that were evaluated to be failures
<row_output>
{{row_output}}
</row_output>

### Evaluation or Scorer data and metadata

<evaluation_evaluation_or_scorer_data>
{{evaluation_evaluation_or_scorer_data}}
</evaluation_evaluation_or_scorer_data>

## Proposed list of available failure categories


<proposed_failure_categories>
{{proposed_failure_categories}}
</proposed_failure_categories>

Does the eval failure you can see above fall into any of the proposed failure categories?
"""

    class CategoryReview(BaseModel):
        """A task failure category."""

        thinking: str = Field(
            description="A detailed reasoning process behind the selection of the category \
    name, description and notes."
        )
        candidate_categories_appropriate: bool = Field(
            description="Whether the proposed failure categories are appropriate for the eval failure."
        )
        new_category_proposal: str | None = Field(
            description="If the proposed failure categories are not appropriate, return a new \
    category that is appropriate for this particular eval failure."
        )

    async def category_review(
        user_context: str,
        row_input: str,
        row_output: str,
        evaluation_evaluation_or_scorer_data: str,
        proposed_failure_categories: list[str],
    ) -> CategoryReview:
        category_review_llm = Agent(
            name="Final Classification",
            instructions=CATEGORIZATION_REVIEW_SYSTEM_PROMPT,
            model=LitellmModel(model=model, api_key=api_key),
            output_type=CategoryReview,
        )

        category_review_prompt_str = CATEGORIZATION_REVIEW_PROMPT.format(
            user_context=user_context,
            row_input=row_input,
            row_output=row_output,
            evaluation_evaluation_or_scorer_data=evaluation_evaluation_or_scorer_data,
            proposed_failure_categories=review_data["task_failure_categories"],
        )

        category_review_result = await Runner.run(
            category_review_llm, category_review_prompt_str
        )

        return category_review_result.final_output

    review_tasks = [
        category_review(
            user_context=user_context,
            row_input=trace_entry["input"],
            row_output=trace_entry["output"],
            evaluation_evaluation_or_scorer_data=trace_entry["scores"],
            proposed_failure_categories=review_data["task_failure_categories"],
        )
        for trace_entry in trace_data
    ]

    category_review_results = await asyncio.gather(*review_tasks)

    # Pretty print the final result (using the first result as an example)
    if category_review_results:
        first_result = category_review_results[0]
        final_result_panel = Panel(
            f"""[bold cyan]Thinking:[/bold cyan] {first_result.thinking}...
[bold cyan]Candidate Categories Appropriate:[/bold cyan] {first_result.candidate_categories_appropriate}
[bold cyan]New Category Proposal:[/bold cyan] {first_result.new_category_proposal}""",
            title="First Test Categorization Result",
            border_style="magenta",
        )
    else:
        final_result_panel = Panel(
            "[red]No category review results available[/red]",
            title="First Test Categorization Result",
            border_style="red",
        )
    console.print(final_result_panel)

    console.print("[bold green]✅ Pipeline completed successfully![/bold green]")


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
    api_key = os.getenv("GOOGLE_API_KEY")

    args: Args = simple_parsing.parse(Args)

    if not user_context_str:
        raise ValueError(
            "User context is required. Please provide a user context about the AI system and what they are trying to evaluate."
        )

    asyncio.run(
        run_pipeline(
            eval_id=eval_id,
            user_context=user_context_str,
            debug=args.debug,
            model=model,
            api_key=args.api_key,
            config_file=args.config_file,
            force_column_selection=args.force_column_selection,
        )
    )
