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

print("Script started!")  # Debug print

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

# Import the focused scorers - ARI and Category-Discovery F1
from evaluation.fails_eval.scorers import (
    create_scorer_suite,
    get_scorer_descriptions,
    AdjustedRand,
    CategoryDiscoveryF1
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
print(f"Parsed args: {args}")  # Debug print

if args.debug:
    args.model = "gemini/gemini-2.5-flash"
    print("Set to debug mode")  # Debug print


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
    
    async def predict(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """
        Predict failure category for a single evaluation failure.
        
        Args:
            inputs: Dict containing row_input, row_output, and evaluation_data
            
        Returns:
            Dict with failure_category, categorization_reason, and category_definition
        """
        
        # Extract inputs from dictionary
        row_input = inputs.get("row_input", "")
        row_output = inputs.get("row_output", "")  
        evaluation_data = inputs.get("evaluation_data", "")
        
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
        
        # For now, use simple pattern-based categorization to test dataset-level scoring
        # This simulates what the full pipeline would return
        transcript = row_input.lower()
        reasoning_text = row_output.lower()
        
        # Simple pattern-based categorization for testing the scorers
        if "format" in transcript or "structure" in reasoning_text or "json" in reasoning_text:
            category = "format_error"
            reason = "Output appears to have formatting or structure issues"
            definition = "Issues with the structure, format, or schema of the output"
        elif "logic" in transcript or "reasoning" in reasoning_text or "contradiction" in reasoning_text:
            category = "logic_error" 
            reason = "Output shows flawed logical reasoning or contradictions"
            definition = "Errors in logical reasoning, inference, or consistency"
        elif "halluc" in transcript or "false" in reasoning_text or "inaccurate" in reasoning_text:
            category = "hallucination"
            reason = "Output contains false, fabricated, or inaccurate information"
            definition = "Model generated false, nonsensical, or fabricated information"
        elif "incomplete" in reasoning_text or "partial" in reasoning_text or "cut off" in reasoning_text:
            category = "incomplete_response"
            reason = "Output is incomplete, partial, or was cut off"
            definition = "Response is incomplete, truncated, or missing required information"
        elif "understand" in reasoning_text or "misinterpreted" in reasoning_text:
            category = "misunderstanding"
            reason = "Model misunderstood the task, input, or requirements"
            definition = "Model failed to correctly understand the task or input context"
        else:
            category = "other"
            reason = "Failure does not fit into standard categories"
            definition = "Miscellaneous failure that doesn't match defined categories"
        
        return {
            "failure_category": category,
            "categorization_reason": reason,
            "category_definition": definition
        }
    
    def _predict_sync(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Synchronous version of predict for debugging."""
        # Extract inputs from dictionary
        row_input = inputs.get("row_input", "")
        row_output = inputs.get("row_output", "")  
        evaluation_data = inputs.get("evaluation_data", "")
        
        # Simple pattern-based categorization for testing the scorers
        transcript = row_input.lower()
        reasoning_text = row_output.lower()
        
        # Simple pattern-based categorization for testing the scorers
        if "format" in transcript or "structure" in reasoning_text or "json" in reasoning_text:
            category = "format_error"
            reason = "Output appears to have formatting or structure issues"
            definition = "Issues with the structure, format, or schema of the output"
        elif "logic" in transcript or "reasoning" in reasoning_text or "contradiction" in reasoning_text:
            category = "logic_error" 
            reason = "Output shows flawed logical reasoning or contradictions"
            definition = "Errors in logical reasoning, inference, or consistency"
        elif "halluc" in transcript or "false" in reasoning_text or "inaccurate" in reasoning_text:
            category = "hallucination"
            reason = "Output contains false, fabricated, or inaccurate information"
            definition = "Model generated false, nonsensical, or fabricated information"
        elif "incomplete" in reasoning_text or "partial" in reasoning_text or "cut off" in reasoning_text:
            category = "incomplete_response"
            reason = "Output is incomplete, partial, or was cut off"
            definition = "Response is incomplete, truncated, or missing required information"
        elif "understand" in reasoning_text or "misinterpreted" in reasoning_text:
            category = "misunderstanding"
            reason = "Model misunderstood the task, input, or requirements"
            definition = "Model failed to correctly understand the task or input context"
        else:
            category = "other"
            reason = "Failure does not fit into standard categories"
            definition = "Miscellaneous failure that doesn't match defined categories"
        
        return {
            "failure_category": category,
            "categorization_reason": reason,
            "category_definition": definition
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
    print("Initializing Weave...")
    weave.init(args.project)
    print(f"Successfully initialized Weave project: {args.project}")
    
    # Load the real dataset
    try:
        print(f"Loading dataset: {args.dataset_ref}")
        dataset = weave.ref(args.dataset_ref).get()
        print(f"Successfully loaded dataset '{args.dataset_ref}' with {len(dataset.rows)} rows")
        
        # Log dataset structure for debugging
        if dataset.rows:
            sample_row = dataset.rows[0]
            print(f"Sample dataset row keys: {list(sample_row.keys())}")
            
    except Exception as e:
        print(f"Failed to load dataset '{args.dataset_ref}': {e}")
        return None
    
    # Initialize the model
    model = FailureCategorizationModel(
        model_name=args.model
    )
    print(f"Initialized model: {args.model}")
    print(f"Model object: {model}")
    print(f"Model predict method: {model.predict}")
    
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
            print(f"Found {len(allowed_labels)} unique ground truth labels: {allowed_labels}")
    
    print("Using dataset-level sophisticated scorers:")
    print("  - Adjusted Rand Index: Partition agreement without caring about label strings")
    print("  - Category-Discovery F1: Coverage × precision with Hungarian assignment")
    
    # Create evaluation logger for dataset-level scoring
    eval_logger = weave.EvaluationLogger(
        model=model,
        dataset=dataset,
        name="failure_categorization_dataset_eval"
    )
    
    # Store all predictions and ground truth for dataset-level scoring
    all_predictions = []
    all_ground_truths = []
    
    print("Starting failure categorization evaluation with dataset-level metrics...")
    print("This will collect all predictions first, then compute sophisticated metrics...")
    
    try:
        print("About to start processing examples...")
        # Process each example and collect predictions
        for i, example in enumerate(dataset.rows):
            print(f"Processing example {i+1}/{len(dataset.rows)}...")
            
            # Prepare model inputs
            model_inputs = {
                "row_input": example.get("example.full_transcript", ""),
                "row_output": example.get("output.reasoning", ""),
                "evaluation_data": example.get("failure_category_reason_gt", "")
            }
            
            # Get model prediction using sync version (since we had issues with async)
            try:
                prediction = model._predict_sync(model_inputs)
                print(f"Prediction result: {prediction}")
            except Exception as e:
                print(f"Prediction failed: {e}")
                # Use fallback prediction for testing
                prediction = {
                    "failure_category": "other",
                    "categorization_reason": "Model prediction failed",
                    "category_definition": "Fallback category"
                }
            
            # Log the prediction with Weave
            try:
                pred_logger = eval_logger.log_prediction(
                    inputs=model_inputs,
                    output=prediction
                )
                pred_logger.finish()
            except Exception as e:
                print(f"Failed to log prediction to Weave: {e}")
            
            # Store for dataset-level metrics
            predicted_category = prediction.get("failure_category", "")
            ground_truth_category = example.get("failure_category_gt", "")
            
            all_predictions.append(predicted_category)
            all_ground_truths.append(ground_truth_category)
            
            print(f"  Predicted: {predicted_category}")
            print(f"  Ground Truth: {ground_truth_category}")
        
        print("All predictions collected. Computing dataset-level metrics...")
        
        # Import the sophisticated scoring functions
        from evaluation.fails_eval.scorers import _best_match_map
        from sklearn.metrics import adjusted_rand_score
        
        # Calculate ARI score (dataset-level partition agreement)
        ari_score = adjusted_rand_score(all_ground_truths, all_predictions)
        
        # Calculate Category-Discovery F1 with Hungarian assignment
        mapping = _best_match_map(all_predictions, all_ground_truths)
        
        # Calculate metrics
        unique_preds = set(all_predictions)
        unique_golds = set(all_ground_truths)
        
        # Precision: % of predicted categories that have valid mappings
        mapped_preds = set(mapping.keys())
        precision = len(mapped_preds) / len(unique_preds) if unique_preds else 0.0
        
        # Coverage/Recall: % of gold categories that were discovered
        mapped_golds = set(mapping.values())
        coverage = len(mapped_golds) / len(unique_golds) if unique_golds else 0.0
        
        # F1: harmonic mean
        category_f1 = 2 * precision * coverage / (precision + coverage + 1e-12)
        
        # Create summary results
        results = {
            "adjusted_rand_index": ari_score,
            "category_discovery_f1": category_f1,
            "precision": precision,
            "coverage": coverage,
            "num_examples": len(all_predictions),
            "unique_predicted_categories": len(unique_preds),
            "unique_ground_truth_categories": len(unique_golds),
            "category_mapping": dict(mapping)
        }
        
        # Log summary with Weave
        try:
            eval_logger.log_summary(results)
            print("Successfully logged results summary to Weave")
        except Exception as e:
            print(f"Failed to log summary to Weave: {e}")
        
        print("Dataset-level evaluation completed successfully!")
        print("\\nDataset-Level Results:")
        print(f"  Adjusted Rand Index: {ari_score:.4f} (partition agreement, -1 to 1, higher better)")
        print(f"  Category-Discovery F1: {category_f1:.4f} (coverage × precision, 0 to 1, higher better)")
        print(f"  Precision: {precision:.4f} (% of predicted categories that are valid)")
        print(f"  Coverage: {coverage:.4f} (% of ground truth categories discovered)")
        print(f"  Examples processed: {len(all_predictions)}")
        print(f"  Unique predicted categories: {len(unique_preds)}")
        print(f"  Unique ground truth categories: {len(unique_golds)}")
        print(f"  Hungarian assignment mapping: {dict(mapping)}")
        
        return results
    except Exception as e:
        print(f"Meta evaluation failed: {e}")
        import traceback
        print(traceback.format_exc())
        return None


async def main():
    """Main function for running the evaluation."""
    print("Main function started!")  # Debug print
    
    if not args.run_eval:
        logger.info("Use --run_eval to run the meta evaluation")
        logger.info("Example: python -m fails.failure_categorization_eval --run_eval")
        return
    
    print("About to check API key...")  # Debug print
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("API key is required. Set GOOGLE_API_KEY environment variable")
        return
    print("API key check passed!")  # Debug print
    
    print("="*60)
    print("FAILURE CATEGORIZATION META EVALUATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset_ref}")
    print(f"Debug mode: {args.debug}")
    print(f"Project: {args.project}")
    print("="*60)
    
    # Run the evaluation
    results = await run_failure_categorization_evaluation()
    
    if results:
        print("="*60)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("="*60)
        print("Check the results above for detailed analysis")
    else:
        print("="*60)
        print("EVALUATION FAILED")
        print("="*60)
        print("Check the logs above for error details")


if __name__ == "__main__":
    asyncio.run(main()) 