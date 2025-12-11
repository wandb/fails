from typing import List

from pydantic import BaseModel, Field

TRACE_PATTERN_DEFINITION = """A trace represents a single execution of a system that has been selected for analysis. \
Traces may be selected because they exhibit specific behaviors, contain errors, fail certain criteria, or match other \
filter conditions. Each trace may exhibit issues or patterns such as:

- Incorrect outputs or behaviors
- Formatting problems or structured output errors
- Execution errors or exceptions
- Too many tool calls or infinite loops
- Edge cases or unusual behaviors
- Performance issues or timeouts
- etc.
"""

FIRST_PASS_CATEGORIZATION_SYSTEM_PROMPT = f"""
# Task - Trace Pattern Categorization

Your task is to output a draft set of notes and candidate pattern categories given trace data from a user's AI system. \
We are trying to help a user understand key failure modes and behavioral patterns in their AI system.

## Trace Definition

{TRACE_PATTERN_DEFINITION}

### How Your Notes and Candidate Pattern Categories Will Be Used

With this rough draft of pattern categories and notes for 1 or a small number of traces, a later step in this pipeline \
will subsequently compare the draft notes and candidate pattern categories across a larger number of traces. \
From here, we will iteratively align and refine the notes and candidate pattern categories until we \
have a set of pattern categories that are consistent across a larger number of traces.

## Inspiration - Open Coding

This task is similar to open coding, where we are trying to identify the underlying issue and phenomenon:

> Open coding attempts to codify, name or classifying the observed phenomenon and is achieved by segmenting \
data into meaningful expressions and describing that data with a single word or short sequence of words

Some examples of open coding questions to consider when drafting the notes and candidate pattern categories:

- Identify the underlying issue and phenomenon *(What?*)
- Identify the phenomenon's attributes *(What kind?*)
- Determine the time, course and location of the behavior *(When? How long? Where?)*
- Identify the intensity of the issue (*How much? How long?*)
- Identify the reasons attached to the issue (*Why?*)
- Identify intention or purpose of the behavior (*Why?)*

Take inspiration from the above open coding questions but there is no need to be exhaustive if it's not relevant \
to the trace data in question.

## Notes on User-provided Data

### Human Annotations

If human annotations are provided, these represent real observations from human reviewers. Pay close attention to \
annotated error categories or issues, as these are valuable signals for identifying patterns.

### Automated Scores

If automated scores are provided (e.g., from scorers or evaluators), use these as indicators but don't treat them \
as absolute truth. Scores can have false positives or false negatives. Look at the actual trace data to understand \
what's really happening.

### LLM-generated Reasoning

Be cautious if the trace data includes 'thinking', 'reasoning' or 'notes' sections that may have come from an LLM. \
These should not be treated as absolute truth. You can still use clearly correct insights, just be cautious about \
trusting them 100%.
"""

FIRST_PASS_CATEGORIZATION_PROMPT = """
Given the specific task context from the user as well as the trace data, please make your best \
guess at the notes and candidate pattern categories for the given trace.

## User context about their AI system

Below is the context from the user about their AI system and what they are trying to analyze. This will help \
you better understand what the user is trying to achieve with their AI system.

<user_context>
{user_context}
</user_context>

{human_annotations_section}

## Trace Data

### Inputs that were given to the system
<trace_input>
{trace_input}
</trace_input>

### Outputs from the system
<trace_output>
{trace_output}
</trace_output>

### Additional Metadata (scores, timestamps, etc.)

<trace_metadata>
{trace_metadata}
</trace_metadata>
{execution_trace_section}
## Analyze and Draft Notes and Candidate Pattern Categories

With the above user context and trace data, please output a draft set of notes and candidate \
pattern categories for the given trace.

### Specificity of Candidate Pattern Categories

Try and be as specific as possible in your categorizations without literally incorporating every single detail of the \
single trace given. The goal is to identify patterns that will likely appear across multiple traces.

### Style of Candidate Pattern Categories

Ensure that the candidate pattern categories are:
- concise and to the point
- lowercase
- separated by underscores
- no more than 5 words maximum
"""


class FirstPassCategory(BaseModel):
    """A first pass categorization of a single trace."""

    category_name: str = Field(description="The name of the category.")
    category_description: str = Field(
        description="A high-level, generic, short description and justification for the category."
    )
    trace_note: str = Field(
        description="A sentence or two of notes specific to what was observed in this individual trace."
    )


class FirstPassCategorization(BaseModel):
    """First pass classification of a single trace."""

    thinking: str = Field(
        description="A detailed thinking process of the classification."
    )
    first_pass_categories: list[FirstPassCategory] = Field(
        description="A short list of 1-3 first pass categories for the trace."
    )


class FirstPassCategorizationResult(FirstPassCategorization):
    """First pass classification of a single trace."""

    trace_id: str = Field(description="The ID of the trace that was classified.")


# ----------------- Clustering draft categorizations -----------------

MAX_N_PATTERN_CATEGORIES = 7

CLUSTERING_SYSTEM_PROMPT = f"""# Task - Clustering Draft Categorizations

Given {{num_traces}} of draft categorizations and notes for a set of traces, cluster \
the categorizations and notes into a defined set of pattern categories.

## Definition - Trace Patterns

{TRACE_PATTERN_DEFINITION}

## Task Context - Clustering Draft Categorizations

The purpose of this task is to examine draft categorizations and notes for a set of traces and cluster the \
categories into a canonical set of pattern categories. The aim is to find a set of pattern categories that \
are consistent across a large number of traces, ideally we have no more than \
{MAX_N_PATTERN_CATEGORIES} pattern categories.

These categories should be specific enough to:
1. Clearly identify failure modes or problematic behaviors
2. Highlight distinct behavioral patterns that could be monitored
3. Be potentially converted into automated scorers for ongoing monitoring

If a trace doesn't fit into any of the defined pattern categories, it should be classified as "other".

Keep all category names lowercase, concise and separated by '_'.
"""

CLUSTERING_PROMPT = f"""
## Draft Categorizations and Notes

Here are the draft categorizations and notes for {{num_traces}} traces:

<draft_categorizations_and_notes>

{{draft_categorizations_and_notes}}
</draft_categorizations_and_notes>

## Output

Output a list of maximum {MAX_N_PATTERN_CATEGORIES} pattern categories - you can output less than \
{MAX_N_PATTERN_CATEGORIES} if you think that's appropriate.
"""

## ----------------- Step 3 - Category Review -----------------


class Category(BaseModel):
    """A pattern category."""

    thinking: str = Field(
        description="A detailed reasoning process behind the selection of the category \
name, description and notes."
    )
    pattern_category_name: str = Field(
        description="""The name of the pattern category. Keep all category \
names lowercase, concise and separated by '_'. If a trace doesn't fit into any of the defined pattern \
categories, it should be classified as 'other'."""
    )
    pattern_category_definition: str = Field(
        description="A short definition of the pattern category."
    )
    pattern_category_notes: str = Field(
        description="A sentence or two of notes for the pattern category."
    )


class ClusteringCategories(BaseModel):
    """Clustering of draft categorizations and notes into a set of pattern categories."""

    category_long_list_thinking: str = Field(
        description="A detailed reasoning process and final decision making \
for the selection of the pattern categories."
    )
    pattern_categories: list[Category] = Field(
        description="""A list of pattern categories. \
If a trace doesn't fit into any of the defined pattern categories, it should be classified as "other"."""
    )


# ----------------- Step 3 - Final Classification -----------------

FINAL_CLASSIFICATION_SYSTEM_PROMPT = """
# Task - Final Classification of Traces

You are a helpful assistant that classifies traces into predefined pattern categories.

Your task is to analyze a single trace and classify it into one of the provided pattern categories.

## Important Notes:
- You must select exactly ONE category from the provided list
- If the trace doesn't clearly fit into any of the predefined categories, classify it as "other"
- Base your classification on the actual trace data, not on assumptions
- Consider the user context to better understand the nature of the behavior or issue
"""

FINAL_CLASSIFICATION_PROMPT = """
Given the following trace data and the list of available pattern categories, \
classify this specific trace into the most appropriate category.

## User Context

<user_context>
{user_context}
</user_context>

{human_annotations_section}

## Trace Data

### Inputs that were given to the system
<trace_input>
{trace_input}
</trace_input>

### Outputs from the system
<trace_output>
{trace_output}
</trace_output>

### Additional Metadata (scores, timestamps, etc.)
<trace_metadata>
{trace_metadata}
</trace_metadata>
{execution_trace_section}
## Available Pattern Categories

<available_pattern_categories>
{available_pattern_categories}
</available_pattern_categories>

## Task

Analyze the above trace and classify it into ONE of the available categories. \
If none of the categories are appropriate, classify it as "other".
"""


class FinalClassification(BaseModel):
    """Final classification of a single trace into predefined categories."""

    thinking: str = Field(
        description="A detailed reasoning process explaining why this specific trace \
belongs to the selected category. Consider the trace characteristics, the category \
definitions, and why this is the best match among all available categories."
    )
    pattern_category: str = Field(
        description="The selected category name from the available pattern categories. \
Must be one of the provided category names or 'other'."
    )
    categorization_reason: str = Field(
        description="Brief notes explaining any specific aspects of this trace \
that influenced the classification decision."
    )


class FinalClassificationResult(FinalClassification):
    """Final classification result with trace ID."""

    trace_id: str = Field(description="The ID of the trace that was classified.")


# -----------------------------------------------------
class PipelineResult(BaseModel):
    pattern_categories: List[Category]
    classifications: List[FinalClassificationResult]
    report: str = ""


# =============================================================================
# Deep Trace Analysis Prompts
# =============================================================================

# Execution trace section - conditionally included when deep_trace_analysis=True
EXECUTION_TRACE_SECTION = """
### Agent Execution Trace

The trace below shows the internal execution flow including tool calls, LLM operations, and timings.

Look for these failure patterns:
- **Tool Use**: Wrong tool, bad parameters, ignored outputs, redundant calls
- **Planning**: Loops, poor ordering, abandoned plans

<agent_execution_trace>
{execution_trace}
</agent_execution_trace>
"""

# Trace compaction prompts
TRACE_COMPACTION_SYSTEM_PROMPT = """You are compacting an agent execution trace for failure analysis.

Your task is to summarize this execution trace concisely while preserving critical information.

PRESERVE (keep exactly as-is or with minimal reduction):
- All tool call names and their key input parameters
- Errors, exceptions, or failure indicators
- Key decision points and reasoning from LLM calls
- Final outputs and results
- The hierarchical structure of the trace

SUMMARIZE/TRUNCATE:
- Verbose intermediate LLM outputs (keep just key decisions)
- Large data payloads in outputs (summarize what type of data)
- Redundant or repetitive information
- Long lists or arrays (indicate count and type)

Output a condensed version of the trace that maintains the tree structure but is more compact."""

TRACE_COMPACTION_USER_PROMPT = """Compact this agent execution trace to approximately {target_tokens} tokens:

{trace_tree}

Output the compacted trace maintaining the tree structure."""
