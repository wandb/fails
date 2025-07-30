#!/usr/bin/env python3
"""
Meta Evaluation for Failure Categorization

This module provides a standalone evaluation system for assessing the performance
of the real failure categorization pipeline. It uses sophisticated dataset-level 
metrics to evaluate how well the pipeline categorizes evaluation failures.

The evaluation uses the complete 3-step pipeline:
1. Draft categorization - Generate initial failure categories  
2. Clustering/Review - Aggregate and refine categories
3. Final classification - Assign failures to refined categories

Usage:
    python -m evaluation.fails_eval.failure_categorization_eval --run_eval
    python -m evaluation.fails_eval.failure_categorization_eval --run_eval --debug
"""

print("Script started!")  # Debug print

import asyncio
import json
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
from rich.console import Console

# Load environment variables
load_dotenv()

# Disable tracing for agents to avoid conflicts
set_tracing_disabled(True)

# Import from the real pipeline
from fails.console import logger
from fails.weave_query import query_evaluation_data, TraceDepth
from fails.pipeline import run_pipeline
from fails.prompts import (
    FIRST_PASS_CATEGORIZATION_SYSTEM_PROMPT,
    FIRST_PASS_CATEGORIZATION_PROMPT,
    Category,
    FirstPassCategorizationResult,
    FinalClassificationResult,
    PipelineResult
)

# Import the focused scorers - ARI and Category-Discovery F1
from evaluation.fails_eval.scorers import (
    create_scorer_suite,
    get_scorer_descriptions,
    AdjustedRand,
    CategoryDiscoveryF1,
    _best_match_map
)

# Disable litellm logging
litellm.turn_off_message_logging = True

# Set up API key for the pipeline
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    os.environ["LLM_API_KEY"] = api_key


@dataclass
class EvalArgs:
    """Arguments for the meta evaluation."""
    
    model: str = "gemini/gemini-2.5-pro"
    debug: bool = False
    run_eval: bool = False
    dataset_ref: str = "speaker_classification_failure_annotation:v0"
    project: str = "wandb-applied-ai-team/eval-failures"
    user_context: str = """
## User AI System Context

My app is trying to identify insights from transcripts of meetings between prospects and our sales team.

## User Eval Context 

To classify speaker IDs from a transcript into whether they are from our company or are a customer/prospect.
"""


# Parse command line arguments
args: EvalArgs = simple_parsing.parse(EvalArgs)
print(f"Parsed args: {args}")  # Debug print

if args.debug:
    args.model = "gemini/gemini-2.5-flash"
    print("Set to debug mode")  # Debug print


class FailureCategorizationModel(weave.Model):
    """
    Weave model for categorizing evaluation failures using the real pipeline.
    
    This model uses the complete 3-step failure categorization pipeline to
    provide realistic evaluation results.
    """
    
    model_name: str
    user_context: str
    
    async def predict(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """
        Predict failure category using the real pipeline.
        
        Args:
            inputs: Dict containing row_input, row_output, and evaluation_data
            
        Returns:
            Dict with failure_category, categorization_reason, and category_definition
        """
        
        # Extract inputs from dictionary
        row_input = inputs.get("row_input", "")
        row_output = inputs.get("row_output", "")  
        evaluation_data = inputs.get("evaluation_data", "")
        
        # Prepare trace data in the format expected by the pipeline
        # The pipeline expects structured dictionaries, not raw strings
        trace_data = [{
            "id": "single_trace_eval",
            "inputs": {
                "full_transcript": row_input,
                "evaluation_context": evaluation_data
            },
            "output": {
                "reasoning": row_output,
                "evaluation_result": "failed"
            },
            "scores": {
                "evaluation_failed": True,
                "context": evaluation_data
            }
        }]
        
        try:
            # Use the real pipeline to categorize the failure  
            console = Console()
            pipeline_result = await run_pipeline(
                trace_data=trace_data,
                user_context=self.user_context,
                model=self.model_name,
                max_concurrent_llm_calls=1,  # Single prediction
                debug=False,  # Keep quiet for individual predictions
                console=console
            )
            
            # Extract the classification result
            if pipeline_result and pipeline_result.classifications:
                classification = pipeline_result.classifications[0]
                
                # Find the category definition from the discovered categories
                category_definition = "Failure category discovered by the pipeline"
                for category in pipeline_result.failure_categories:
                    if category.failure_category_name == classification.failure_category:
                        category_definition = category.failure_category_definition
                        break
                
                return {
                    "failure_category": classification.failure_category,
                    "categorization_reason": classification.categorization_reason,
                    "category_definition": category_definition
                }
            else:
                # Fallback if no classification was produced
                return {
                    "failure_category": "other",
                    "categorization_reason": "Pipeline did not produce a classification",
                    "category_definition": "Fallback category for unclassified failures"
                }
                
        except Exception as e:
            print(f"Pipeline prediction failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback prediction
            return {
                "failure_category": "other",
                "categorization_reason": f"Pipeline execution failed: {str(e)}",
                "category_definition": "Fallback category due to pipeline error"
            }
    
    def _predict_sync(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Synchronous version of predict for compatibility."""
        return asyncio.run(self.predict(inputs))


# ----------------- Main Evaluation Functions -----------------

async def run_failure_categorization_evaluation():
    """
    Run the meta evaluation for failure categorization using the real pipeline.
    
    This function:
    1. Loads the annotated dataset from Weave
    2. Prepares all examples as batch data for the pipeline
    3. Runs the complete pipeline on all examples at once
    4. Extracts individual predictions and computes sophisticated dataset-level metrics
    5. Returns the evaluation results
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
    
    # Initialize the real pipeline model (for Weave logging)
    model = FailureCategorizationModel(
        model_name=args.model,
        user_context=args.user_context
    )
    print(f"Initialized real pipeline model: {args.model}")
    
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
    
    print("Using sophisticated dataset-level metrics:")
    print("  - Adjusted Rand Index: Measures partition agreement ignoring label names")
    print("  - Category-Discovery F1: Combines coverage × precision with Hungarian assignment")
    print("  - Hungarian Assignment: Maps predicted categories to ground truth via Jaccard overlap")
    
    # Create evaluation logger for dataset-level scoring
    eval_logger = weave.EvaluationLogger(
        model=model,
        dataset=dataset,
        name="failure_categorization_real_pipeline_eval"
    )
    
    print("Starting failure categorization evaluation with the REAL PIPELINE...")
    print("Processing ALL examples together through the complete 3-step pipeline (draft → clustering → classification)")
    
    try:
        # Prepare ALL examples as batch trace data for the pipeline
        all_trace_data = []
        example_to_trace_mapping = {}  # Maps trace_id to example index
        
        for i, example in enumerate(dataset.rows):
            trace_id = f"eval_example_{i}"
            
            # Prepare trace data in the format expected by the pipeline
            trace_entry = {
                "id": trace_id,
                "inputs": {
                    "full_transcript": example.get("example.full_transcript", ""),
                    "evaluation_context": example.get("failure_category_reason_gt", "")
                },
                "output": {
                    "reasoning": example.get("output.reasoning", ""),
                    "evaluation_result": "failed"
                },
                "scores": {
                    "evaluation_failed": True,
                    "context": example.get("failure_category_reason_gt", "")
                }
            }
            
            all_trace_data.append(trace_entry)
            example_to_trace_mapping[trace_id] = i
        
        print(f"Prepared {len(all_trace_data)} examples for batch pipeline processing...")
        
        # Run the real pipeline on ALL examples at once
        console = Console()
        pipeline_result = await run_pipeline(
            trace_data=all_trace_data,
            user_context=args.user_context,
            model=args.model,
            max_concurrent_llm_calls=5,  # Allow some concurrency for batch processing
            debug=False,  # Keep quiet for batch processing
            console=console
        )
        
        print(f"Pipeline completed! Processing results...")
        
        # Store all predictions and ground truth for dataset-level scoring
        all_predictions = []
        all_ground_truths = []
        
        if pipeline_result and pipeline_result.classifications:
            print(f"Got {len(pipeline_result.classifications)} classifications from pipeline")
            
            # Process each classification result
            for classification in pipeline_result.classifications:
                trace_id = classification.trace_id
                example_idx = example_to_trace_mapping.get(trace_id)
                
                if example_idx is not None:
                    example = dataset.rows[example_idx]
                    
                    # Find the category definition from the discovered categories
                    category_definition = "Failure category discovered by the pipeline"
                    for category in pipeline_result.failure_categories:
                        if category.failure_category_name == classification.failure_category:
                            category_definition = category.failure_category_definition
                            break
                    
                    # Create prediction output for logging
                    model_inputs = {
                        "row_input": example.get("example.full_transcript", ""),
                        "row_output": example.get("output.reasoning", ""),
                        "evaluation_data": example.get("failure_category_reason_gt", "")
                    }
                    
                    prediction = {
                        "failure_category": classification.failure_category,
                        "categorization_reason": classification.categorization_reason,
                        "category_definition": category_definition
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
                    predicted_category = classification.failure_category
                    ground_truth_category = example.get("failure_category_gt", "")
                    
                    all_predictions.append(predicted_category)
                    all_ground_truths.append(ground_truth_category)
                    
                    print(f"Example {example_idx+1}: Predicted='{predicted_category}', Ground Truth='{ground_truth_category}'")
                    
        else:
            print("Pipeline returned no classifications! Using fallback predictions...")
            # Fallback: use "other" for all predictions
            for i, example in enumerate(dataset.rows):
                all_predictions.append("other")
                all_ground_truths.append(example.get("failure_category_gt", ""))
        
        print("All predictions collected from real pipeline. Computing dataset-level metrics...")
        
        # Import sklearn for ARI calculation
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
            "category_mapping": dict(mapping),
            "model_used": args.model,
            "pipeline_type": "real_3_step_pipeline_batch",
            "discovered_categories": [cat.failure_category_name for cat in pipeline_result.failure_categories] if pipeline_result else []
        }
        
        # Log summary with Weave
        try:
            eval_logger.log_summary(results)
            print("Successfully logged results summary to Weave")
        except Exception as e:
            print(f"Failed to log summary to Weave: {e}")
        
        print("="*60)
        print("REAL PIPELINE META EVALUATION COMPLETED!")
        print("="*60)
        print(f"  Adjusted Rand Index: {ari_score:.4f} (partition agreement, -1 to 1, higher better)")
        print(f"  Category-Discovery F1: {category_f1:.4f} (coverage × precision, 0 to 1, higher better)")
        print(f"  Precision: {precision:.4f} (% of predicted categories that are valid)")
        print(f"  Coverage: {coverage:.4f} (% of ground truth categories discovered)")
        print(f"  Examples processed: {len(all_predictions)}")
        print(f"  Unique predicted categories: {len(unique_preds)}")
        print(f"  Unique ground truth categories: {len(unique_golds)}")
        print(f"  Hungarian assignment mapping: {dict(mapping)}")
        print(f"  Model: {args.model}")
        if pipeline_result:
            print(f"  Pipeline discovered categories: {[cat.failure_category_name for cat in pipeline_result.failure_categories]}")
        print("="*60)
        
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
        logger.info("Example: python -m evaluation.fails_eval.failure_categorization_eval --run_eval")
        return
    
    print("About to check API key...")  # Debug print
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("API key is required. Set GOOGLE_API_KEY environment variable")
        return
    print("API key check passed!")  # Debug print
    
    print("="*60)
    print("REAL FAILURE CATEGORIZATION PIPELINE META EVALUATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset_ref}")
    print(f"Debug mode: {args.debug}")
    print(f"Project: {args.project}")
    print("Pipeline: REAL 3-STEP PIPELINE (draft → clustering → classification)")
    print("="*60)
    
    # Run the evaluation
    results = await run_failure_categorization_evaluation()
    
    if results:
        print("="*60)
        print("REAL PIPELINE EVALUATION COMPLETED SUCCESSFULLY")
        print("="*60)
        print("View detailed results at: https://wandb.ai/wandb-applied-ai-team/eval-failures/weave")
    else:
        print("="*60)
        print("REAL PIPELINE EVALUATION FAILED")
        print("="*60)
        print("Check the logs above for error details")


if __name__ == "__main__":
    asyncio.run(main()) 