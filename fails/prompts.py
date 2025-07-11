from pydantic import BaseModel, Field


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
{row_input}
</row_input>

### Outputs that were evaluated to be failures
<row_output>
{row_output}
</row_output>

### Evaluation or Scorer data and metadata

<evaluation_evaluation_or_scorer_data>
{evaluation_evaluation_or_scorer_data}
</evaluation_evaluation_or_scorer_data>

## Analyse

With the above user context and evaluation failure data, please output a draft set of notes and candidate \
task failure categories for the given row input and row output. 

### User-provided eval reasoning

Be cautious if the users eval data has provided a 'thinking', 'reasoning' or 'notion' section that has come form a LLM. \
These 'reasons' for the eval decision should not be treated as absolute truth as the LLM can still possibly be hallucinating \
or making up reasons for the eval decision. You can still use this data, just be cautious.

"""

class FirstPassCategory(BaseModel):
    """A first pass categorization of a single evaluation failure."""

    category_name: str = Field(description="The name of the category.")
    category_description: str = Field(description="A high-level, generic, short description and justification for the category.")
    eval_failure_note: str = Field(description="A sentence or two of notes sepcific to what was observed in this individual evaluation failure.")

class FirstPassCategorization(BaseModel):
    """First pass classification of a single evaluation failure."""

    thinking: str = Field(
        description="A detailed thinking process of the classification."
    )
    first_pass_categories: list[FirstPassCategory] = Field(
        description="A short list of 1-3 first pass categories for the evaluation failure."
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

## ----------------- Step 3 - Category Review -----------------

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


# ----------------- Step 3 - Final Classification -----------------

FINAL_CLASSIFICATION_SYSTEM_PROMPT = """
# Task - Final Classification of Evaluation Failures

You are a helpful assistant that classifies evaluation failures into predefined categories.

Your task is to analyze a single evaluation failure and classify it into one of the provided task failure categories.

## Important Notes:
- You must select exactly ONE category from the provided list
- If the failure doesn't clearly fit into any of the predefined categories, classify it as "other"
- Base your classification on the actual failure data, not on assumptions
- Consider the user context to better understand the nature of the failure
"""

FINAL_CLASSIFICATION_PROMPT = """
Given the following evaluation failure data and the list of available failure categories, \
classify this specific failure into the most appropriate category.

## User Context

<user_context>
{user_context}
</user_context>

## Evaluation Failure Data

### Inputs that were given to the system
<row_input>
{row_input}
</row_input>

### Outputs that were evaluated to be failures
<row_output>
{row_output}
</row_output>

### Evaluation or Scorer data and metadata
<evaluation_evaluation_or_scorer_data>
{evaluation_evaluation_or_scorer_data}
</evaluation_evaluation_or_scorer_data>

## Available Failure Categories

<available_failure_categories>
{available_failure_categories}
</available_failure_categories>

## Task

Analyze the above evaluation failure and classify it into ONE of the available categories. \
If none of the categories are appropriate, classify it as "other".
"""

class FinalClassification(BaseModel):
    """Final classification of a single evaluation failure into predefined categories."""
    
    thinking: str = Field(
        description="A detailed reasoning process explaining why this specific failure \
belongs to the selected category. Consider the failure characteristics, the category \
definitions, and why this is the best match among all available categories."
    )
    selected_category: str = Field(
        description="The selected category name from the available categories. \
Must be one of the provided category names or 'other'."
    )
    confidence_score: float = Field(
        description="A confidence score between 0.0 and 1.0 indicating how well \
this failure fits the selected category.",
        ge=0.0,
        le=1.0
    )
    classification_notes: str = Field(
        description="Brief notes explaining any specific aspects of this failure \
that influenced the classification decision."
    )


class FinalClassificationResult(FinalClassification):
    """Final classification result with trace ID."""
    
    trace_id: str = Field(description="The ID of the trace that was classified.")


# Legacy models kept for backwards compatibility
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
{row_input}
</row_input>

### Outputs that were evaluated to be failures
<row_output>
{row_output}
</row_output>

### Evaluation or Scorer data and metadata

<evaluation_evaluation_or_scorer_data>
{evaluation_evaluation_or_scorer_data}
</evaluation_evaluation_or_scorer_data>

## Proposed list of available failure categories


<proposed_failure_categories>
{proposed_failure_categories}
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
    new_category_proposal: Category | None = Field(
        description="If the proposed failure categories are not appropriate, return a new \
category that is appropriate for this particular eval failure."
    )

# -----------------------------------------------------