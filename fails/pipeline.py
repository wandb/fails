import asyncio
import json
import logging
import os
import sys
from asyncio import Semaphore
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import re
from datetime import datetime

import litellm
import simple_parsing
import weave
import yaml
from agents import Agent, Runner, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Import wandb for report creation
import wandb

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fails.cli.column_context_selector import simple_arrow_selection
from fails.cli.config_selector import select_config
from fails.cli.context_collector import collect_user_context
from fails.cli.evaluation_selector import interactive_evaluation_selection
from fails.cli.failure_selector import interactive_failure_column_selection
from fails.cli.header import get_fails_header_for_rich
from fails.prompts import (
    CLUSTERING_PROMPT,
    CLUSTERING_SYSTEM_PROMPT,
    FINAL_CLASSIFICATION_PROMPT,
    FINAL_CLASSIFICATION_SYSTEM_PROMPT,
    FIRST_PASS_CATEGORIZATION_PROMPT,
    FIRST_PASS_CATEGORIZATION_SYSTEM_PROMPT,
    Category,
    ClusteringCategories,
    FinalClassification,
    FinalClassificationResult,
    FirstPassCategorization,
    FirstPassCategorizationResult,
    PipelineResult,
)
from fails.utils import (
    display_evaluation_summary,
    generate_evaluation_report,
    prepare_trace_data_for_pipeline,
    validate_failure_column,
)
from fails.spinner import FailsSpinner
from fails.weave_query import (
    TraceDepth,
    filter_evaluation_data_columns,
    get_available_columns,
    query_evaluation_data,
)

load_dotenv()
set_tracing_disabled(True)

logging.getLogger("LiteLLM").setLevel(logging.ERROR)
litellm.turn_off_message_logging = True


@weave.op
def save_report_to_file(
    report_text: str,
    eval_name: str,
    wandb_entity: str,
    wandb_project: str,
    console: Console,
) -> Optional[str]:
    """
    Save the evaluation report to a local markdown file.
    
    Args:
        report_text: The markdown report text
        eval_name: Name of the evaluation
        wandb_entity: W&B entity name
        wandb_project: W&B project name
        console: Rich console for output
    
    Returns:
        Path to the saved file, or None if save fails
    """
    try:
        # Create reports directory if it doesn't exist
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        # Clean eval_name for filename (remove special characters)
        clean_eval_name = re.sub(r'[^\w\s-]', '', eval_name)
        clean_eval_name = re.sub(r'[-\s]+', '-', clean_eval_name)
        
        filename = f"{timestamp}_{wandb_entity}_{wandb_project}_{clean_eval_name}.md"
        filepath = reports_dir / filename
        
        # Write the report to file
        with open(filepath, 'w', encoding='utf-8') as f:
            # Add metadata header
            f.write(f"---\n")
            f.write(f"title: {eval_name}\n")
            f.write(f"entity: {wandb_entity}\n")
            f.write(f"project: {wandb_project}\n")
            f.write(f"generated: {datetime.now().isoformat()}\n")
            f.write(f"---\n\n")
            
            # Write the report content
            f.write(report_text)
        
        return str(filepath)
        
    except Exception as e:
        console.print(f"[yellow]Warning: Could not save report to file: {e}[/yellow]")
        return None


@weave.op
def create_wandb_report(
    entity_name: str,
    project_name: str,
    title: str,
    markdown_report_text: str,
    description: Optional[str] = None,
) -> Optional[str]:
    """
    Create a W&B report from the evaluation failure analysis.
    
    Args:
        entity_name: The W&B entity (team or username)
        project_name: The W&B project name
        title: Title of the W&B Report
        markdown_report_text: Markdown text for the report body
        description: Optional description of the W&B Report
    
    Returns:
        The URL of the created report, or None if creation fails
    """
    try:
        # Only import wandb_workspaces if we're actually creating a report
        try:
            import wandb_workspaces.reports.v2 as wr
        except ImportError:
            console = Console()
            console.print("[yellow]wandb_workspaces not installed. Install with: pip install wandb-workspaces[/yellow]")
            return None
        
        # Initialize wandb
        wandb.init(
            entity=entity_name, 
            project=project_name, 
            job_type="fails_report_creation",
            reinit=True  # Allow reinit since we already have a wandb session
        )
        
        # Initialize the report
        report = wr.Report(
            entity=entity_name,
            project=project_name,
            title=title,
            description=description or f"Evaluation failure analysis for {project_name}",
            width="fluid",
        )
        
        # Parse markdown content into W&B blocks
        blocks = []
        lines = markdown_report_text.strip().split("\n")
        current_paragraph = []
        
        for line in lines:
            # Check for headers
            h1_match = re.match(r"^# (.+)$", line)
            h2_match = re.match(r"^## (.+)$", line)
            h3_match = re.match(r"^### (.+)$", line)
            
            # If we hit a header and have paragraph content, finalize the paragraph
            if (h1_match or h2_match or h3_match) and current_paragraph:
                blocks.append(wr.P("\n".join(current_paragraph)))
                current_paragraph = []
            
            # Handle the current line
            if h1_match:
                blocks.append(wr.H1(h1_match.group(1)))
            elif h2_match:
                blocks.append(wr.H2(h2_match.group(1)))
            elif h3_match:
                blocks.append(wr.H3(h3_match.group(1)))
            else:
                if line.strip():  # Only add non-empty lines
                    current_paragraph.append(line)
        
        # Don't forget any remaining paragraph content
        if current_paragraph:
            blocks.append(wr.P("\n".join(current_paragraph)))
        
        # Set the blocks
        report.blocks = blocks
        
        # Save the report
        report.save()
        wandb.finish()
        
        return report.url
        
    except Exception as e:
        console = Console()
        console.print(f"[yellow]Warning: Could not create W&B report: {e}[/yellow]")
        return None


def get_api_key_for_model(model: str) -> str:
    """Get the appropriate API key based on the model provider.
    
    LiteLLM uses format: provider/model-name
    """
    provider = model.split('/')[0].lower() if '/' in model else 'openai'
    
    # Map provider prefixes to environment variable names
    provider_env_map = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'claude': 'ANTHROPIC_API_KEY',
        'gemini': 'GOOGLE_API_KEY',
        'google': 'GOOGLE_API_KEY',
        'vertex_ai': 'GOOGLE_API_KEY',
        'cohere': 'COHERE_API_KEY',
        'replicate': 'REPLICATE_API_KEY',
        'azure': 'AZURE_API_KEY',
    }
    
    env_var = provider_env_map.get(provider, 'OPENAI_API_KEY')
    api_key = os.getenv(env_var)
    
    if not api_key:
        raise ValueError(f"{env_var} environment variable not set for model {model}")
    
    return api_key


@dataclass
class Args:
    """Script arguments for the pipeline."""

    model: str | None = None  # Will default to LLM_MODEL env var or fallback value
    debug: bool = False
    force_eval_select: bool = False  # Force selection of evaluation URL and columns
    config_file: str = "./config/failure_categorization_config.yaml"
    test_config: str | None = None  # Optional test config file
    wandb_entity: str | None = None  # Will be extracted from URL or provided via CLI
    wandb_project: str | None = None  # Will be extracted from URL or provided via CLI
    wandb_logging_entity: str | None = None  # Optional entity for weave.init() - can be from CLI or .env
    wandb_logging_project: str = "eval-failures"  # Project for weave.init() - can be from CLI or .env
    n_samples: int | None = None
    max_concurrent_llm_calls: int = 20  # Control concurrent LLM API calls
    eval_id: str | None = None  # Optional evaluation ID to skip interactive selection


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
    config_file: str, wandb_entity: str, wandb_project: str, columns: List[str], console: Console
) -> None:
    """
    Save column preferences for a specific project to config file.

    Args:
        config_file: Path to the config file
        wandb_entity: Weave entity name
        wandb_project: Weave project name
        columns: List of column names to save
        console: Rich console instance
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

    # Don't print here - will be shown in Configuration Summary


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
        Dict with 'failure_column' and 'failure_filter' if preferences exist, None otherwise
        The 'failure_filter' can be either a direct value or a dict with operator
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
        ):
            result = {
                "failure_column": config[project_key]["failure_column"],
            }
            
            # Handle both old and new format
            if "failure_filter" in config[project_key]:
                # New format with operator support
                result["failure_filter"] = config[project_key]["failure_filter"]
            elif "failure_value" in config[project_key]:
                # Old format - convert to new format
                result["failure_filter"] = {"$eq": config[project_key]["failure_value"]}
            else:
                return None
                
            return result

        return None

    except (yaml.YAMLError, KeyError) as e:
        print(f"[yellow]Warning: Error reading config file: {e}[/yellow]")
        return None


@weave.op
def load_user_context_preferences(
    config_file: str, wandb_entity: str, wandb_project: str
) -> Optional[Tuple[str, str]]:
    """
    Load user context preferences for a specific project from config file.

    Args:
        config_file: Path to the config file
        wandb_entity: Weave entity name
        wandb_project: Weave project name

    Returns:
        Tuple of (system_context, eval_context) if preferences exist, None otherwise
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
            and "user_ai_system_context" in config[project_key]
            and "user_eval_context" in config[project_key]
        ):
            return (
                config[project_key]["user_ai_system_context"],
                config[project_key]["user_eval_context"]
            )

        return None

    except (yaml.YAMLError, KeyError) as e:
        print(f"[yellow]Warning: Error reading config file: {e}[/yellow]")
        return None


@weave.op
def save_user_context_preferences(
    config_file: str, wandb_entity: str, wandb_project: str, 
    system_context: str, eval_context: str, console: Console
) -> None:
    """
    Save user context preferences for a specific project to config file.

    Args:
        config_file: Path to the config file
        wandb_entity: Weave entity name
        wandb_project: Weave project name
        system_context: AI system context
        eval_context: Evaluation context
        console: Rich console instance
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

    config[project_key]["user_ai_system_context"] = system_context
    config[project_key]["user_eval_context"] = eval_context

    # Save config as YAML
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    console.print(f"\n[bright_magenta]✓[/bright_magenta] Saved project config to: {config_file}")
    console.print("\n")


@weave.op
def save_failure_column_preferences(
    config_file: str,
    wandb_entity: str,
    wandb_project: str,
    failure_column: str,
    failure_value: Any,
    console: Console,
) -> None:
    """
    Save failure column preferences for a specific project to config file.

    Args:
        config_file: Path to the config file
        wandb_entity: Weave entity name
        wandb_project: Weave project name
        failure_column: The column name to filter by
        failure_value: The filter value (can be a direct value or dict with operator)
        console: Rich console instance
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
    
    # Save the filter value in a consistent format
    if isinstance(failure_value, dict):
        # It's already in operator format
        config[project_key]["failure_filter"] = failure_value
    else:
        # Direct value means equals operator
        config[project_key]["failure_filter"] = {"$eq": failure_value}

    # Save config as YAML
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Don't print here - will be shown in Configuration Summary


@weave.op
def get_column_preferences(
    config_file: str,
    wandb_entity: str,
    wandb_project: str,
    eval_id: str,
    console: Console,
    force_eval_select: bool = False,
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
        force_eval_select: Force re-selection of columns
        debug: Debug mode

    Returns:
        Tuple of (failure_config, columns_list) where:
        - failure_config: Dict with 'failure_column' and 'failure_filter' or None
        - columns_list: List of column names for data extraction
    """
    # First, handle failure column selection - no need to print header

    # Get all available columns first
    column_info = get_available_columns(
        eval_id=eval_id,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        include_nested_paths=True,
        max_nesting_depth=4,
    )
    all_columns = column_info["all_columns"]
    
    # Extract sample values from the sample trace
    sample_values = {}
    if "sample_trace" in column_info:
        sample_trace = column_info["sample_trace"]
        for col in all_columns:
            # Navigate through the nested structure
            value = sample_trace
            try:
                for key in col.split('.'):
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        value = None
                        break
                if value is not None:
                    sample_values[col] = [value]  # Store as list for consistency
            except (AttributeError, TypeError):
                pass

    # Check for saved failure column preferences
    saved_failure_config = load_failure_column_preferences(
        config_file, wandb_entity, wandb_project
    )
    failure_config = None

    if saved_failure_config and not force_eval_select:
        # Use saved failure column preferences
        filter_desc = ""
        failure_filter = saved_failure_config.get('failure_filter', {})
        
        if isinstance(failure_filter, dict) and len(failure_filter) == 1:
            op_key = list(failure_filter.keys())[0]
            op_value = failure_filter[op_key]
            
            # Find operator symbol
            from fails.cli.failure_selector import FailureColumnSelector
            op_symbol = op_key
            for op_name, op_info in FailureColumnSelector.OPERATORS.items():
                if op_info["key"] == op_key:
                    op_symbol = op_info["symbol"]
                    break
            
            if op_key == "$eq":
                filter_desc = f"== {op_value}"
            else:
                filter_desc = f"{op_symbol} {op_value}"
        
        # Store the failure config but don't print panel yet
        failure_config = saved_failure_config
    else:
        # Perform failure column selection
        # Perform failure column selection silently

        # Interactive failure column selection
        failure_column, failure_value = interactive_failure_column_selection(
            console, all_columns, sample_values
        )
        
        # Handle user cancellation
        if failure_column is None:
            console.print("[red]Exiting: Failure column selection was cancelled.[/red]")
            return None, None

        if failure_column and failure_value is not None:
            # Build the failure config
            failure_config = {
                "failure_column": failure_column,
            }
            
            # Handle the filter format
            if isinstance(failure_value, dict):
                # Already in operator format
                failure_config["failure_filter"] = failure_value
            else:
                # Direct value means equals
                failure_config["failure_filter"] = {"$eq": failure_value}

            # Save the failure column preferences
            save_failure_column_preferences(
                config_file, wandb_entity, wandb_project, failure_column, failure_value, console
            )
        else:
            console.print(
                "[yellow]No failure column selected - will process all traces[/yellow]"
            )

    # Now handle regular column selection
    saved_columns = load_column_preferences(config_file, wandb_entity, wandb_project)

    if saved_columns and not force_eval_select:
        # Use saved preferences
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

        # Build combined content for both failure filter and column configuration
        combined_content = ""
        
        # Add failure filter section if we have saved failure config
        if saved_failure_config and not force_eval_select:
            filter_desc = ""
            failure_filter = saved_failure_config.get('failure_filter', {})
            
            if isinstance(failure_filter, dict) and len(failure_filter) == 1:
                op_key = list(failure_filter.keys())[0]
                op_value = failure_filter[op_key]
                
                # Find operator symbol
                from fails.cli.failure_selector import FailureColumnSelector
                op_symbol = op_key
                for op_name, op_info in FailureColumnSelector.OPERATORS.items():
                    if op_info["key"] == op_key:
                        op_symbol = op_info["symbol"]
                        break
                
                if op_key == "$eq":
                    filter_desc = f"== {op_value}"
                else:
                    filter_desc = f"{op_symbol} {op_value}"
            
            combined_content += "[bold]Failure Filter[/bold]\n"
            combined_content += f"Filter: [cyan]{saved_failure_config['failure_column']}[/cyan] {filter_desc}\n\n"
        
        # Add column configuration section
        combined_content += "[bold]Context Columns[/bold]\n"
        combined_content += f"Selected {len(selected_columns)} columns\n"

        # Add grouped columns to content
        for prefix in sorted(grouped.keys()):
            cols = grouped[prefix]
            combined_content += f"\n[yellow]{prefix}[/yellow] ({len(cols)} columns):\n"
            for col in cols[:3]:
                combined_content += f"  • {col}\n"
            if len(cols) > 3:
                combined_content += f"  ... and {len(cols) - 3} more\n"

        if other_cols:
            combined_content += (
                f"\n[yellow]Top-level[/yellow] ({len(other_cols)} columns):\n"
            )
            for col in other_cols:
                combined_content += f"  • {col}\n"

        # Display everything in one merged panel
        console.print(
            Panel(
                combined_content.rstrip(),
                title="Configuration Summary",
                border_style="yellow",
                padding=(1, 2),
            )
        )

    else:
        # Perform column selection - no need to print header

        # Debug messages removed for cleaner UI

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
            and not any(
                part.startswith("_") for part in col.split(".")
            )  # Exclude underscore-prefixed keys
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
        

        # Save preferences silently
        save_column_preferences(
            config_file, wandb_entity, wandb_project, list(selected_columns), console
        )

        # Build and display combined configuration panel for new selection
        combined_content = ""
        
        # Add failure filter section if one was just selected
        if failure_config:
            filter_desc = ""
            failure_filter = failure_config.get('failure_filter', {})
            
            if isinstance(failure_filter, dict) and len(failure_filter) == 1:
                op_key = list(failure_filter.keys())[0]
                op_value = failure_filter[op_key]
                
                # Find operator symbol
                from fails.cli.failure_selector import FailureColumnSelector
                op_symbol = op_key
                for op_name, op_info in FailureColumnSelector.OPERATORS.items():
                    if op_info["key"] == op_key:
                        op_symbol = op_info["symbol"]
                        break
                
                if op_key == "$eq":
                    filter_desc = f"== {op_value}"
                else:
                    filter_desc = f"{op_symbol} {op_value}"
            
            combined_content += "[bold]Failure Filter[/bold]\n"
            combined_content += f"Filter: [cyan]{failure_config['failure_column']} {filter_desc}[/cyan]\n\n"
        
        # Add column configuration section
        combined_content += "[bold]Context Columns[/bold]\n"
        combined_content += f"Selected {len(selected_columns)} columns\n"
        
        # Group columns for display
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
        
        # Add grouped columns to content
        for prefix in sorted(grouped.keys()):
            cols = grouped[prefix]
            combined_content += f"\n[bright_magenta]{prefix}[/bright_magenta] ({len(cols)} columns):\n"
            for col in cols[:5]:  # Show up to 5 columns
                combined_content += f"  • {col}\n"
            if len(cols) > 5:
                combined_content += f"  ... and {len(cols) - 5} more\n"
        
        if other_cols:
            combined_content += f"\n[bright_magenta]Top-level[/bright_magenta] ({len(other_cols)} columns):\n"
            for col in other_cols:
                combined_content += f"  • {col}\n"
        
        # Display the combined panel
        console.print(
            Panel(
                combined_content.rstrip(),
                title="Configuration Summary",
                border_style="white",
                padding=(1, 2),
            )
        )

        if debug:
            console.print("\n[dim]Selected columns:[/dim]")
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
        evaluation_evaluation_or_scorer_data = json.dumps(
            evaluation_evaluation_or_scorer_data, indent=2
        )

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
    llm_semaphore: Semaphore,
    debug: bool = False,
) -> FirstPassCategorizationResult:
    async with llm_semaphore:
        draft_categorization_llm = Agent(
            name="Row by Row",
            instructions=FIRST_PASS_CATEGORIZATION_SYSTEM_PROMPT,
            model=LitellmModel(model=model, api_key=get_api_key_for_model(model)),
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
    max_concurrent_llm_calls: int,
    debug: bool = False,
) -> list[FirstPassCategorizationResult]:
    # Create shared semaphore for all draft categorization tasks
    llm_semaphore = Semaphore(max_concurrent_llm_calls)
    
    # Create tasks with trace_id and shared semaphore
    tasks = [
        draft_categorization(
            trace_id=trace_entry["id"],
            row_input=trace_entry["inputs"],
            row_output=trace_entry["output"],
            evaluation_evaluation_or_scorer_data=trace_entry["scores"],
            user_context=user_context,
            model=model,
            debug=debug,
            llm_semaphore=llm_semaphore,
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
        evaluation_evaluation_or_scorer_data = json.dumps(
            evaluation_evaluation_or_scorer_data, indent=2
        )

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
    llm_semaphore: Semaphore,
) -> FinalClassificationResult:
    async with llm_semaphore:
        final_classification_llm = Agent(
            name="Final Classification",
            instructions=FINAL_CLASSIFICATION_SYSTEM_PROMPT,
            model=LitellmModel(model=model, api_key=get_api_key_for_model(model)),
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

    # Add the trace_id to the classification result
    final_classification_result = FinalClassificationResult(
        trace_id=trace_id,
        thinking=classification_result.final_output.thinking,
        failure_category=classification_result.final_output.failure_category,
        categorization_reason=classification_result.final_output.categorization_reason,
    )

    return final_classification_result


@weave.op
async def run_final_classification(
    trace_data: list[dict],
    user_context: str,
    available_categories_str: str,
    model: str,
    max_concurrent_llm_calls: int,
    debug: bool = False,
) -> list[FinalClassificationResult]:
    # Create shared semaphore for all final classification tasks
    llm_semaphore = Semaphore(max_concurrent_llm_calls)
    
    classification_tasks = [
        final_classification(
            trace_id=trace_entry["id"],
            row_input=trace_entry["inputs"],
            row_output=trace_entry["output"],
            evaluation_evaluation_or_scorer_data=trace_entry["scores"],
            user_context=user_context,
            available_categories_str=available_categories_str,
            model=model,
            llm_semaphore=llm_semaphore,
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
    max_concurrent_llm_calls: int,
) -> ClusteringCategories:
    llm_semaphore = Semaphore(max_concurrent_llm_calls)
    async with llm_semaphore:  # Control concurrent LLM calls
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
            model=LitellmModel(model=model, api_key=get_api_key_for_model(model)),
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
    max_concurrent_llm_calls: int,
    debug: bool = False,
    console: Console = Console(),
) -> PipelineResult:
    # ----------------- STEP 1: Draft categorization -----------------
    console.print("\n[bold cyan]Step 1: Draft Categorization[/bold cyan]")
    console.print(f"[bright_magenta]  Starting draft categorization for {len(trace_data)} traces...[/bright_magenta]")

    if debug:
        first_pass_categorization_prompt_str = (
            construct_first_pass_categorization_prompt(
                row_input=trace_data[0]["inputs"],
                row_output=trace_data[0]["output"],
                evaluation_evaluation_or_scorer_data=trace_data[0]["scores"],
                user_context=user_context,
            )
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

    # Start spinner for draft categorization
    draft_spinner = FailsSpinner(f"Categorizing {len(trace_data)} failure traces")
    draft_spinner.start()
    
    draft_categorization_results = await run_draft_categorization(
        trace_data=trace_data,
        user_context=user_context,
        model=model,
        max_concurrent_llm_calls=max_concurrent_llm_calls,
        debug=debug,
    )

    num_draft_categorizations = len(draft_categorization_results)
    
    draft_spinner.stop(f"Completed {num_draft_categorizations} draft categorizations", success=True)

    # ----------------- STEP 2: Review categorizations -----------------

    console.print("\n[bold cyan]Step 2: Review & Clustering[/bold cyan]")

    # Create resclustering prompt (needed for the agent)
    draft_categorization_results_str = "\n" + "=" * 80 + "\n"

    unique_candidate_categories = set()
    for c_i, draft_categorization_result in enumerate(draft_categorization_results):
        draft_categorization_results_str += f"### Evaluation Trace ID:\nTrace ID: {draft_categorization_result.trace_id}\n\n"

        if debug:
            result_table = Table(show_header=True, box=None, padding=(0, 1))
            result_table.add_column("Candidate Category", style="cyan", width=120)
            result_table.add_column("Category Description", style="white", width=120)
            result_table.add_column("Eval Failure Note", style="dim", width=120)

        for i, first_pass_category in enumerate(
            draft_categorization_result.first_pass_categories
        ):
            draft_categorization_results_str += f"#### Candidate Category Name {i + 1}\n\n`{first_pass_category.category_name}`\n\n"
            draft_categorization_results_str += f"#### Category Description {i + 1}:\n\n{first_pass_category.category_description}\n\n"
            draft_categorization_results_str += f"#### Eval Failure Note {i + 1}\n\n{first_pass_category.eval_failure_note}\n\n"
            
            unique_candidate_categories.add(first_pass_category.category_name)
            
            if debug:
                result_table.add_row(
                    first_pass_category.category_name,
                    first_pass_category.category_description,
                    first_pass_category.eval_failure_note,
                )

        draft_categorization_results_str += "\n" + "=" * 80 + "\n"
        console.print(f"[bright_magenta]  Reviewing and clustering {len(unique_candidate_categories)} candidate categories...[/bright_magenta]")
        if debug:
            console.print(
                Panel(
                    result_table,
                    title=f"[green]Trace {c_i + 1}[/green]",
                    border_style="green",
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

    # Start spinner for review
    review_spinner = FailsSpinner("Clustering and reviewing failure categories")
    review_spinner.start()
    
    review_data = await aggregate_categorizations(
        draft_categorization_results_str=draft_categorization_results_str,
        num_draft_categorizations=num_draft_categorizations,
        user_context=user_context,
        model=model,
        max_concurrent_llm_calls=max_concurrent_llm_calls,
    )
    
    review_spinner.stop("  ✓ Review completed successfully!", success=True)

    if debug:
        console.print("=" * 80)
        console.print("Candidate categories:")
        console.print("-" * 80)
        console.print(review_data.category_long_list_thinking)
        console.print("-" * 80)
        for category in review_data.task_failure_categories:
            console.print(f"Category: {category.failure_category_name}")
            console.print(f"Description: {category.failure_category_definition}")
            console.print(f"Notes: {category.failure_category_notes}")
            console.print("-" * 80)

    # ----------------- STEP 3: Final classification -----------------

    console.print("\n[bold cyan]Step 3: Final Classification[/bold cyan]")
    console.print("[bright_magenta]  Performing final classification of failures...[/bright_magenta]")

    # Add "other" category to the list
    all_categories = review_data.task_failure_categories + [
        Category(
            thinking="This is the default category for failures that don't fit into any other category",
            failure_category_name="other",
            failure_category_definition="Can be used if the evaluation failure sample can't be classified into one of the other classes",
            failure_category_notes="Default category for unclassifiable failures",
        )
    ]

    # Format categories for the prompt
    categories_str = ""
    for i, category in enumerate(all_categories):
        categories_str += f"\n### Category {i + 1}: {category.failure_category_name}\n"
        categories_str += f"**Description:** {category.failure_category_definition}\n"
        categories_str += f"**Notes:** {category.failure_category_notes}\n"

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

    # Start spinner for final classification
    classification_spinner = FailsSpinner(f"Classifying {len(trace_data)} failures into categories")
    classification_spinner.start()
    
    classification_results_per_trace = await run_final_classification(
        trace_data=trace_data,
        user_context=user_context,
        available_categories_str=categories_str,
        model=model,
        max_concurrent_llm_calls=max_concurrent_llm_calls,
        debug=debug,
    )
    
    classification_spinner.stop("Classification complete", success=True)

    return PipelineResult(
        failure_categories=all_categories,
        classifications=classification_results_per_trace,
    )



@weave.op
async def run_extract_and_classify_pipeline(
    eval_id: str,
    user_context: str,
    debug: bool,
    model: str,
    max_concurrent_llm_calls: int,
    config_file_path: str,
    wandb_entity: str,
    wandb_project: str,
    force_eval_select: bool = False,
    n_samples: int | None = None,
) -> PipelineResult:
    # Query Weave for evaluation data using the enhanced API
    console = Console()

    if debug:
        console.print(
            f"[bold red]🔍 DEBUG MODE ENABLED, switching to {model}[/bold red]"
        )
        model = "gemini/gemini-2.5-flash"
        n_samples = 3 if n_samples is None else n_samples

    # Display user context in a nice box
    if debug:
        console.print(
            Panel(
                user_context,
                title="User Context",
                border_style="white",
                padding=(1, 2),
                width=105,
            )
        )
    
    # ----------------- Column Selection -----------------

    # Check for saved column preferences
    failure_config, columns_for_query = get_column_preferences(
        config_file=config_file_path,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        eval_id=eval_id,
        debug=debug,
        force_eval_select=force_eval_select,
        console=console,
    )

    # Start spinner for data fetching
    data_spinner = FailsSpinner("Querying evaluation data")
    data_spinner.start()
    
    eval_data = query_evaluation_data(
        eval_id=eval_id,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        columns=columns_for_query,  # Use the selected columns + display_name
        include_outputs=True,
        deep_ref_extraction=False,
        trace_depth=TraceDepth.DIRECT_CHILDREN,  # Get evaluation + direct children
        include_hierarchy=True,
        limit=n_samples,
        filter_dict={failure_config["failure_column"]: failure_config["failure_filter"]}
        if failure_config
        else None,
    )
    
    data_spinner.stop("Evaluation data retrieved", success=True)

    # ----------------- Data Processing -----------------
    # Filter the evaluation data to only include selected columns
    if debug:
        console.print(
            "[bold cyan]🔍 Filtering evaluation data to only include selected columns...[/bold cyan]"
        )
    eval_data = filter_evaluation_data_columns(eval_data, columns_for_query)

    # Display evaluation summary
    display_evaluation_summary(eval_data, failure_config, console)

    # Validate failure column if one was selected
    if failure_config:
        validate_failure_column(eval_data, failure_config, console)

    # Prepare trace data for pipeline
    trace_data = prepare_trace_data_for_pipeline(
        eval_data, debug, console, n_samples=n_samples
    )

    console.print("")  # Add spacing before pipeline execution

    # ----------------- Pipeline Execution -----------------
    pipeline_result = await run_pipeline(
        trace_data=trace_data,
        user_context=user_context,
        model=model,
        max_concurrent_llm_calls=max_concurrent_llm_calls,
        debug=debug,
        console=console,
    )
    final_classification_results = pipeline_result.classifications
    all_categories = pipeline_result.failure_categories

    # ----------------- Generate Evaluation Report -----------------

    console.print("\n[bold cyan]Step 4: Report Generation[/bold cyan]")
    console.print("[bright_magenta]  Generating evaluation report...[/bright_magenta]")

    report = generate_evaluation_report(
        final_classification_results=final_classification_results,
        all_categories=all_categories,
        eval_name=eval_data["evaluation"].get("display_name", eval_id),
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
    )

    # Display the report
    console.print("")  # Add spacing
    console.print(Panel(
        report,
        title="Evaluation Failures Report",
        border_style="white",
        padding=(1, 2),
    ))

    # Save report to local file
    eval_name = eval_data["evaluation"].get("display_name", eval_id)
    local_filepath = save_report_to_file(
        report_text=report,
        eval_name=eval_name,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        console=console
    )
    
    # Create W&B report
    # report_url = create_wandb_report(
    #     entity_name=wandb_entity,
    #     project_name=wandb_project,
    #     title=f"Evaluation Failures: {eval_data['evaluation'].get('display_name', eval_id)}",
    #     markdown_report_text=report,
    #     description=f"Failure categorization analysis for evaluation {eval_id}"
    # )
    
    # Display final results in an organized panel
    console.print()  # Add spacing
    
    # Create content for the completion panel
    completion_content = []
    
    if local_filepath:
        completion_content.append("[bold]Local Report[/bold]")
        completion_content.append(f"  [green]✓[/green] Saved to: [cyan]{local_filepath}[/cyan]")
        completion_content.append("")
    
    # if report_url:
    #     completion_content.append("[bold]W&B Report[/bold]")
    #     completion_content.append("  [green]✓[/green] Published successfully")
    #     completion_content.append(f"  📊 View at: [cyan]{report_url}[/cyan]")
    # else:
    #     completion_content.append("[bold]W&B Report[/bold]")
    #     completion_content.append("  [dim]Skipped (wandb-workspaces may not be installed)[/dim]")
    
    # Display in a nice panel
    console.print(Panel(
        "\n".join(completion_content),
        title="[bright_magenta]✓ Pipeline Completed Successfully[/bright_magenta]",
        border_style="bright_magenta",
        padding=(1, 2),
    ))

    pipeline_result.report = report

    if debug:
        console.print(
            Panel(
                pipeline_result.model_dump_json(indent=4),
                title="Full Pipeline Result (debug mode)",
                border_style="blue",
                padding=(1, 1),
            )
        )

    return pipeline_result


if __name__ == "__main__":
    # Create console instance for welcome message
    console = Console()
    
    # Print welcome message using shared header
    console.print(get_fails_header_for_rich())
    
    args: Args = simple_parsing.parse(Args)
    
    # Check for model from LLM_MODEL environment variable if not provided via CLI
    if args.model is None:
        args.model = os.getenv("LLM_MODEL", "gemini/gemini-2.5-pro")  # Default fallback
    
    # Check environment variables for wandb_logging_entity and wandb_logging_project if not provided via CLI
    if args.wandb_logging_entity is None:
        args.wandb_logging_entity = os.getenv("WANDB_LOGGING_ENTITY")
    if not args.wandb_logging_project:
        env_project = os.getenv("WANDB_LOGGING_PROJECT")
        if env_project:
            args.wandb_logging_project = env_project
    
    # Check for n_samples from environment if not provided via CLI
    if args.n_samples is None:
        env_n_samples = os.getenv("N_SAMPLES")
        if env_n_samples:
            try:
                args.n_samples = int(env_n_samples)
            except ValueError:
                console.print(f"[yellow]Warning: Invalid N_SAMPLES value in .env: {env_n_samples}[/yellow]")
    
    # Check for max_concurrent_llm_calls from environment if not provided via CLI
    env_max_concurrent = os.getenv("MAX_CONCURRENT_LLM_CALLS")
    if env_max_concurrent:
        try:
            # Only override if not explicitly set via CLI (check if it's still the default value)
            if args.max_concurrent_llm_calls == 20:  # 20 is the default value
                args.max_concurrent_llm_calls = int(env_max_concurrent)
        except ValueError:
            console.print(f"[yellow]Warning: Invalid MAX_CONCURRENT_LLM_CALLS value in .env: {env_max_concurrent}[/yellow]")
    
    # Load test config if specified
    test_config = None
    if args.test_config:
        with open(args.test_config, 'r') as f:
            test_config = yaml.safe_load(f)
    
    # Determine evaluation ID
    eval_id = args.eval_id  # Command-line arg takes precedence
    
    console = Console()
    
    if not eval_id and test_config and test_config.get('test_mode', {}).get('enabled'):
        # Use test config if in test mode
        if test_config.get('test_mode', {}).get('skip_eval_selection'):
            eval_id = test_config.get('test_evaluation_id')
            console.print(f"[yellow]Test mode: Using evaluation ID from config: {eval_id}[/yellow]")
        
    # Variables to hold extracted components
    wandb_entity_extracted = None
    wandb_project_extracted = None
    selected_config_path = None
    
    # Check if we need to select a configuration
    if not eval_id and not args.force_eval_select and not args.wandb_entity and not args.wandb_project:
        # No explicit config provided, show selector
        console.print("")  # Add space after logo
        config_result = select_config("./config")
        
        if config_result:
            if config_result.get('force_selection'):
                # User chose to create new config OR no configs exist
                args.force_eval_select = True
                if config_result.get('no_configs'):
                    console.print()  # Add spacing before next step
            else:
                # User selected an existing config
                selected_config_path = config_result['filepath']
                wandb_entity_extracted = config_result['entity']
                wandb_project_extracted = config_result['project']
                
                # Force evaluation selection to get the current eval ID
                args.force_eval_select = True
        else:
            # User cancelled (pressed 'q' in the selector)
            console.print("[yellow]Configuration selection cancelled. Exiting.[/yellow]")
            sys.exit(0)
    
    if not eval_id:
        # Interactive selection if --force_eval_select is used or forced from config selector
        if args.force_eval_select:
            try:
                result = interactive_evaluation_selection(console)
                if not result:
                    console.print("[red]No evaluation selected. Exiting.[/red]")
                    sys.exit(1)
                # Extract components from result
                entity, project, eval_id = result
                if entity and project:
                    wandb_entity_extracted = entity
                    wandb_project_extracted = project

            except Exception as e:
                console.print(f"[red]Error during evaluation selection: {str(e)}[/red]")
                console.print("[yellow]Please try again with a valid evaluation URL.[/yellow]")
                sys.exit(1)
    
    # Override settings from test config if available
    if test_config and test_config.get('test_mode', {}).get('enabled'):
        test_settings = test_config.get('test_mode', {})
        if test_settings.get('n_samples') and not args.n_samples:
            args.n_samples = test_settings.get('n_samples')
        if test_settings.get('model') and args.debug:  # Only override model in debug mode
            model = test_settings.get('model')
        else:
            model = args.model
    else:
        model = args.model

    # Use extracted entity/project if available, otherwise use args
    final_wandb_entity = wandb_entity_extracted or args.wandb_entity
    final_wandb_project = wandb_project_extracted or args.wandb_project
    
    # Validate that we have entity and project
    if not final_wandb_entity or not final_wandb_project:
        console.print("[red]Error: W&B entity and project are required.[/red]")
        console.print("[yellow]These should be extracted from the evaluation URL or provided via --wandb-entity and --wandb-project[/yellow]")
        sys.exit(1)
    
    # Use selected config path if available, otherwise construct from entity/project
    if selected_config_path:
        config_file = selected_config_path
    else:
        config_file = f"./config/{final_wandb_entity}_{final_wandb_project}_config.yaml"
    
    # Step 2: Collect or load user context
    saved_context = load_user_context_preferences(config_file, final_wandb_entity, final_wandb_project)
    
    if saved_context and not args.force_eval_select:
        # Use saved user context
        USER_AI_SYSTEM_CONTEXT, USER_EVAL_CONTEXT = saved_context
        console.print("\n[dim]Using saved user context from configuration[/dim]")
    else:
        # Collect user context interactively
        if saved_context and args.force_eval_select:
            default_system, default_eval = saved_context
        else:
            console.print("\n[dim]No saved user context found[/dim]")
            default_system, default_eval = None, None
        
        # Run the context collector
        context_result = collect_user_context(default_system, default_eval)
        
        if not context_result:
            console.print("[red]User context collection cancelled. Exiting.[/red]")
            sys.exit(1)
        
        USER_AI_SYSTEM_CONTEXT, USER_EVAL_CONTEXT = context_result
        
        # Save the user context
        save_user_context_preferences(
            config_file, final_wandb_entity, final_wandb_project,
            USER_AI_SYSTEM_CONTEXT, USER_EVAL_CONTEXT, console
        )
 
    # Update args.config_file to use our resolved config_file path
    args.config_file = config_file

    if args.debug:
        litellm._turn_on_debug()

    # Initialize Weave with optional entity
    # If wandb_logging_entity is provided (via CLI or .env), use entity/project format
    # Otherwise, just use the project name
    if args.wandb_logging_entity:
        weave_project = f"{args.wandb_logging_entity}/{args.wandb_logging_project}"
    else:
        weave_project = args.wandb_logging_project
    
    weave.init(weave_project)

    asyncio.run(
        run_extract_and_classify_pipeline(
            eval_id=eval_id,
            user_context="",
            debug=args.debug,
            model=model,
            max_concurrent_llm_calls=args.max_concurrent_llm_calls,
            config_file_path=args.config_file,
            force_eval_select=args.force_eval_select,
            wandb_entity=final_wandb_entity,
            wandb_project=final_wandb_project,
            n_samples=args.n_samples,
        )
    )
