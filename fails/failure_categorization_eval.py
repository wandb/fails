#!/usr/bin/env python3
"""
Meta Evaluation for Failure Categorization

This module provides a standalone evaluation system for assessing the performance
of failure categorization models. It's designed to be independent from the main
pipeline development.

Usage:
    python -m fails.failure_categorization_eval --run_eval
    python -m fails.failure_categorization_eval --run_eval --debug
"""

import asyncio
import os
import sys
from dataclasses import dataclass
from typing import Dict, Any, List, cast

import litellm
import simple_parsing
import weave
from agents import Agent, Runner, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Disable tracing for agents to avoid conflicts
set_tracing_disabled(True)

# Import logger and weave_query - simplified imports
from fails.console import logger
from fails.weave_query import query_evaluation_data, TraceDepth

# Import the sophisticated scorers - no fallbacks
from fails.scorers import (
    create_scorer_suite,
    get_scorer_descriptions,
    MicroF1,
    AdjustedRand,
    Silhouette,
    DaviesBouldin,
    ClusterEntropy,
    InvalidLabelRate
)

# Disable litellm logging
litellm.turn_off_message_logging = True


@dataclass
class EvalArgs:
    """Arguments for the meta evaluation."""
    
    model: str = "gemini/gemini-2.5-pro"
    debug: bool = False
    run_eval: bool = False
    dataset_ref: str = "speaker_classification_failure_annotation:v0"
    project: str = "wandb-applied-ai-team/eval-failures"


# Parse command line arguments
args: EvalArgs = simple_parsing.parse(EvalArgs)

if args.debug:
    args.model = "gemini/gemini-2.5-flash"


# Evaluation failure definition (imported from pipeline logic)
EVALUATION_FAILURE_DEFINITION = """An evaluation failure is defined as the output of a single row \
that failed the evaluator or scorer criteria. An individual row that failed can evaluation might do so for a \
number of reasons such as:

- The output was judged to be incorrect by the evaluator or scorer
- The output was not formatted correctly.
- The output had a code execution error.
- etc.
"""

# Categorization prompt (adapted from pipeline)
CATEGORIZATION_PROMPT = f"""
# Task - Evaluation Failure Categorization

Output failure category, reasoning, and category definition for the given evaluation failure.

## Task Context

{EVALUATION_FAILURE_DEFINITION}

Your task is to analyze the evaluation failure and provide:
1. A specific failure category (lowercase, underscore-separated)
2. Clear reasoning for why this failure belongs to this category
3. A definition of what this category represents

## Evaluation Failure Data

### Input Data
<row_input>
{{row_input}}
</row_input>

### Output Data (that failed evaluation)
<row_output>
{{row_output}}
</row_output>

### Evaluation/Scorer Information
<evaluation_data>
{{evaluation_data}}
</evaluation_data>

## Instructions

Analyze the failure and provide:
- failure_category: A specific category name (lowercase, underscore-separated)
- categorization_reason: Clear reasoning for this categorization
- category_definition: Definition of what this category represents

Common failure categories include:
- hallucination: Model generates false or nonsensical information
- format_error: Output doesn't match expected format
- incomplete_response: Output is partial or cut off
- misunderstanding: Model misunderstood the task or input
- factual_error: Incorrect factual information
- logic_error: Flawed reasoning or logical inconsistency
- other: Doesn't fit standard categories

Be specific and provide clear reasoning.
"""


class FailureCategorizationOutput(BaseModel):
    """Output format for failure categorization."""
    
    failure_category: str = Field(
        description="The failure category (lowercase, underscore-separated)"
    )
    categorization_reason: str = Field(
        description="Clear reasoning for this categorization"
    )
    category_definition: str = Field(
        description="Definition of what this category represents"
    )


class FailureCategorizationModel(weave.Model):
    """
    Weave model for categorizing evaluation failures.
    
    This model wraps the pipeline's categorization logic and provides
    a consistent interface for evaluation.
    """
    
    model_name: str
    
    @weave.op()
    async def predict(self, row_input: str, row_output: str, evaluation_data: str) -> Dict[str, str]:
        """
        Predict failure category for a single evaluation failure.
        
        Args:
            row_input: The input data that was evaluated
            row_output: The output that failed evaluation
            evaluation_data: The evaluation/scorer data and metadata
            
        Returns:
            Dict with failure_category, categorization_reason, and category_definition
        """
        
        # Read API key from environment - never store as class attribute
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key is None:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        # Create categorization agent
        categorization_agent = Agent(
            name="Failure Categorization",
            instructions=CATEGORIZATION_PROMPT.format(
                row_input=row_input,
                row_output=row_output,
                evaluation_data=evaluation_data,
            ),
            model=LitellmModel(model=self.model_name, api_key=api_key),
            output_type=FailureCategorizationOutput,
        )
        
        # Run categorization
        result = await Runner.run(categorization_agent, "Categorize this evaluation failure")
        
        # Return in the requested format
        return {
            "failure_category": result.final_output.failure_category,
            "categorization_reason": result.final_output.categorization_reason,
            "category_definition": result.final_output.category_definition
        }


# ----------------- Main Evaluation Functions -----------------

async def run_failure_categorization_evaluation():
    """
    Run the meta evaluation for failure categorization.
    
    This function:
    1. Loads the annotated dataset from Weave
    2. Initializes the failure categorization model
    3. Creates and runs the evaluation with sophisticated scorers
    4. Returns the evaluation results
    """
    
    # Initialize weave
    weave.init(args.project)
    logger.info(f"Initialized Weave project: {args.project}")
    
    # Load the dataset
    try:
        dataset = weave.ref(args.dataset_ref).get()
        logger.info(f"Loaded dataset '{args.dataset_ref}' with {len(dataset.rows)} rows")
        
        # Log dataset structure for debugging
        if dataset.rows:
            sample_row = dataset.rows[0]
            logger.info(f"Sample dataset row keys: {list(sample_row.keys())}")
            
    except Exception as e:
        logger.error(f"Failed to load dataset '{args.dataset_ref}': {e}")
        return None
    
    # Initialize the model
    model = FailureCategorizationModel(
        model_name=args.model
    )
    logger.info(f"Initialized model: {args.model}")
    
    # Get allowed labels from dataset if available
    allowed_labels = None
    if dataset.rows:
        # Extract unique ground truth labels for validation
        gt_labels = set()
        for row in dataset.rows:
            if 'ground_truth_category' in row:
                gt_labels.add(row['ground_truth_category'])
            elif 'failure_category_gt' in row:
                gt_labels.add(row['failure_category_gt'])
        
        if gt_labels:
            allowed_labels = list(gt_labels)
            logger.info(f"Found {len(allowed_labels)} unique ground truth labels: {allowed_labels}")
    
    # Create the sophisticated scorer suite
    scorer_dict = create_scorer_suite(allowed_labels)
    scorer_descriptions = get_scorer_descriptions()
    scorer_suite = list(scorer_dict.values())
    
    logger.info("Using sophisticated scorers:")
    for name, description in scorer_descriptions.items():
        logger.info(f"  - {name}: {description}")
    
    # Create preprocessing function to map dataset columns to model inputs
    def preprocess_model_input(example):
        """Map dataset columns to model expected inputs."""
        return {
            "row_input": example.get("example.full_transcript", ""),
            "row_output": example.get("output.reasoning", ""),
            "evaluation_data": example.get("failure_category_reason_gt", "")
        }
    
    # Create the evaluation
    evaluation = weave.Evaluation(
        dataset=dataset,
        scorers=cast(Any, scorer_suite),
        preprocess_model_input=preprocess_model_input,
        name=f"failure_categorization_meta_eval"
    )
    
    # Run the evaluation
    logger.info("Starting failure categorization meta evaluation...")
    logger.info("This may take a while for embedding-based scorers...")
    
    try:
        results = await evaluation.evaluate(model)
        logger.info("Meta evaluation completed successfully!")
        logger.info(f"Results summary: {results}")
        
        # Log detailed results for each scorer
        logger.info("\nDetailed Results:")
        for scorer_name, score in results.items():
            if scorer_name in scorer_descriptions:
                logger.info(f"  {scorer_name}: {score:.4f} - {scorer_descriptions[scorer_name]}")
        
        return results
    except Exception as e:
        logger.error(f"Meta evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


async def main():
    """Main function for running the evaluation."""
    
    if not args.run_eval:
        logger.info("Use --run_eval to run the meta evaluation")
        logger.info("Example: python -m fails.failure_categorization_eval --run_eval")
        return
    
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("API key is required. Set GOOGLE_API_KEY environment variable")
        return
    
    logger.info("="*60)
    logger.info("FAILURE CATEGORIZATION META EVALUATION")
    logger.info("="*60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset_ref}")
    logger.info(f"Debug mode: {args.debug}")
    logger.info(f"Project: {args.project}")
    logger.info("="*60)
    
    # Run the evaluation
    results = await run_failure_categorization_evaluation()
    
    if results:
        logger.info("="*60)
        logger.info("EVALUATION COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info("Check the Weave UI for detailed results and analysis")
    else:
        logger.error("="*60)
        logger.error("EVALUATION FAILED")
        logger.error("="*60)
        logger.error("Check the logs above for error details")


if __name__ == "__main__":
    asyncio.run(main()) 