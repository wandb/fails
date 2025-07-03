import asyncio
import os
from dataclasses import dataclass

import litellm
import simple_parsing
from agents import Agent, Runner, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel
from console import logger
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

set_tracing_disabled(True)

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fails.weave_query import (
    query_evaluation_data,
    TraceDepth,
)

litellm.turn_off_message_logging = True

@dataclass
class Args:
    """Script arguments for the pipeline."""

    user_context: str = "my eval is about finding the weather in Tokyo"
    model: str = "gemini/gemini-2.5-pro"
    api_key: str | None = os.getenv("GOOGLE_API_KEY")
    debug: bool = False


args: Args = simple_parsing.parse(Args)

if args.debug:
    args.model = "gemini/gemini-2.5-flash"


EVALUATION_FAILURE_DEFINITION = """An evaluation failure is defined as the output of a single row \
that failed the evaluator or scorer criteria. An individual row that failed can evaluation might do so for a \
number of reasons such as:

- The output was judged to be incorrect by the evaluator or scorer
- The output was not formatted correctly.
- The output had a code execution error.
- etc.
"""

FIRST_PASS_CATEGORIZATION_PROMPT = f"""
# Task - Evaluation Failure Categorization

Output a draft set of notes and candiate task failure categories given evaluation failures data.

## Task Context - Evaluation Failure Categorization

The purpose of this task is to draft a rough set of notes and candidate task failure categories given \
evaluation failures data. 

{EVALUATION_FAILURE_DEFINITION}

With this rough draft of categories and notesfor 1 or a small number of rows, a later step in this pipeline \
will subsequently compare the draft notes and candidate task failure categories across a larger number of rows. \
From here, we will iteratively align and refine the notes and candidate task failure categories until we \
have a set of notes and candidate task failure categories that are consistent across a larger number of rows.

With this context, please make your best guess at the notes and candidate task failure categories for the \
given row input and row output.

## Evaluation Failure Data

### Inputs 
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

With the above task context and evaluation failure data, please output a draft set of notes and candidate \
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

CLUSTERING_PROMPT = f"""# Task - Clustering Draft Categorizations

Given {{num_traces}} of draft categorizations and notes for a set of evaluation failures, cluster the \
categorizations and notes into a defined set of task failure categories.

## Definition - Evaluation Failure

{EVALUATION_FAILURE_DEFINITION}

## Task Context - Clustering Draft Categorizations

The purpose of this task is examine draft categorizations and notes for a set of evaluation failures and cluster the \
categories into a canonical set of task failure categories. The aim is to find a set of task failure categories that \
are consistent across a large number of evaluation failures, ideally we have no more than \
{MAX_N_TASK_FAILURE_CATEGORIES} eval failure categories.

If a trace doesn't fit into any of the defined task failure categories, it should be classified as "other".

Keep all category names lowercase, concise and separated by '_'.

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


async def main():
    eval_id = "0197a72d-2704-7ced-8c07-0fa1e0ab0557"

    # Query Weave for evaluation data using the enhanced API
    print("Fetching evaluation trace...")

    if args.debug:
        first_pass_trace_limit = 3
    else:
        first_pass_trace_limit = None

    eval_data = query_evaluation_data(
        eval_id=eval_id,
        entity_name="wandb-applied-ai-team",
        project_name="eval-failures",
        trace_depth=TraceDepth.DIRECT_CHILDREN,  # Get evaluation + direct children
        include_hierarchy=True,
        limit=first_pass_trace_limit,
    )

    logger.info(f"Evaluation ID: {eval_data['evaluation']['id']}")
    logger.info(f"Op Name: {eval_data['evaluation'].get('op_name', 'Unknown')}")
    logger.info(f"Total traces: {eval_data['trace_count']['total']}")
    logger.info(f"Direct children: {eval_data['trace_count'].get('direct_children', 0)}")

    # Show evaluation summary if available
    if "summary" in eval_data["evaluation"]:
        logger.info(f"Summary: {eval_data['evaluation']['summary']}")

    logger.info(f"Evaluation data keys: {eval_data.keys()}")
    logger.info(f"Evaluation hierarchy: {eval_data['hierarchy']}")

    logger.info(f"First child keys: {eval_data['children'][0].keys()}\n")
    trace_data = []
    
    for trace in eval_data["children"]:
        if args.debug:
            logger.info(f"Full Trace: {trace}")
            logger.info("*" * 50)
            logger.info(f"Trace ID: {trace['id']}")
            logger.info(f"Trace Name: {trace['display_name']}")
            logger.info(f"Trace Op Name: {trace['op_name']}")
            logger.info(f"Trace Started At: {trace['started_at']}")
            logger.info(f"Trace Ended At: {trace['ended_at']}")
            # print(f"Trace Input: {trace['input']}")
            logger.info(f"Trace Output: {trace['output']}")
            # print(f"Trace Evaluation or Scorer Data: {trace['evaluation_or_scorer_data']}")
            logger.info(f"Trace Summary: {trace['summary']}")
            logger.info(f"Trace Output Ootput: {trace['output']["output"]}")
            logger.info("\n" + "=" * 50 + "\n")

        # trace_data.append(
        #     {
        #         "input": trace["inputs"],
        #         "output": {
        #             "affiliation": trace["output"]["affiliation_score"],
        #             "reasoning": trace["output"]["output"]["reasoning"],
        #         },
        #         "scores": trace["output"]["scores"],
        #     }
        # )

    print("\n" + "=" * 50 + "\n")

    # ----------------- STEP 1: Draft categorization -----------------

    async def draft_categorization(
        trace_id: str,
        row_input: str,
        row_output: str,
        evaluation_evaluation_or_scorer_data: str,
    ) -> FirstPassCategorizationResult:
        draft_categorization_llm = Agent(
            name="Row by Row",
            instructions=FIRST_PASS_CATEGORIZATION_PROMPT.format(
                row_input=row_input,
                row_output=row_output,
                evaluation_evaluation_or_scorer_data=evaluation_evaluation_or_scorer_data,
            ),
            model=LitellmModel(model=args.model, api_key=args.api_key),
            output_type=FirstPassCategorization,
        )

        draft_categorization_result = await Runner.run(
            draft_categorization_llm, "What's the weather in Tokyo?"
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
            trace_id=trace["id"],
            row_input=trace["input"],
            row_output=trace["output"],
            evaluation_evaluation_or_scorer_data=trace["scores"],
        )
        for trace in trace_data
    ]
    draft_categorization_results = await asyncio.gather(*tasks)
    num_draft_categorizations = len(draft_categorization_results)
    draft_categorization_results_str = f"Here are the draft categorizations and notes for {num_draft_categorizations} rows: \n"
    draft_categorization_results_str += "=" * 50 + "\n"

    for draft_categorization_result in draft_categorization_results:
        draft_categorization_results_str += f"Trace ID: {draft_categorization_result.trace_id}\n"
        draft_categorization_results_str += f"Notes: {draft_categorization_result.notes}\n"
        draft_categorization_results_str += f"Candidate Task Failure Categories:\n{draft_categorization_result.candidate_task_failure_categories}\n"
        draft_categorization_results_str += "=" * 50 + "\n"

    logger.info(f"Draft classification results:\n{draft_categorization_results_str}\n")

    # ----------------- STEP 2: Review categorizations -----------------

    review_categorizations_llm = Agent(
        name="Review Agent",
        instructions=CLUSTERING_PROMPT.format(
            num_traces=num_draft_categorizations,
            draft_categorizations_and_notes=draft_categorization_results_str,
        ),
        model=LitellmModel(model=args.model, api_key=args.api_key),
        output_type=ClusteringCategories,
    )

    review_result = await Runner.run(
        review_categorizations_llm, "Review the categorizations"
    )

    logger.info(
        f"Review result:\n{review_result.final_output.model_dump_json(indent=2)}\n"
    )

    draft_classification_llm = Agent(
        name="Final Classification",
        instructions=FIRST_PASS_CATEGORIZATION_PROMPT.format(
            row_input="what is the weather in Tokyo?",
            row_output="The weather in Tokyo is sunny.",
            evaluation_evaluation_or_scorer_data="incorrect",
        ),
        model=LitellmModel(model=args.model, api_key=args.api_key),
        output_type=FirstPassCategorization,
    )

    result = await Runner.run(draft_classification_llm, "What's the weather in Tokyo?")
    logger.info(result.final_output.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
