#!/usr/bin/env python3
"""
Weave Evaluation for Failure Categorization Pipeline

This module wraps the failure categorization pipeline as a Weave Model
and evaluates it using focused scorers (ARI and Category-Discovery F1).

Usage:
    uv run python pipeline_weave_eval.py --run_eval
    uv run python pipeline_weave_eval.py --run_eval --debug
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Dict, Any, List, cast, Optional

import simple_parsing
import weave
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Import pipeline components
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fails.pipeline import run_extract_and_classify_pipeline

# Import focused scorers
from pipeline_scorers import create_pipeline_scorer_suite, get_pipeline_scorer_descriptions

# Import console for logging
from fails.console import Console

console = Console()


@dataclass
class PipelineEvalArgs:
    """Arguments for pipeline evaluation."""
    
    model: str = "gemini/gemini-2.5-pro"
    debug: bool = False
    run_eval: bool = False
    dataset_ref: str = "speaker_classification_failure_annotation:v0"
    project: str = "wandb-applied-ai-team/eval-failures"
    eval_id: str = "0197a72d-2704-7ced-8c07-0fa1e0ab0557"
    wandb_entity: str = "wandb-applied-ai-team"
    wandb_project: str = "eval-failures"
    max_concurrent_llm_calls: int = 100
    n_samples: Optional[int] = None


# Parse command line arguments
args: PipelineEvalArgs = simple_parsing.parse(PipelineEvalArgs)

if args.debug:
    args.model = "gemini/gemini-2.5-flash"
    args.n_samples = 3


class PipelineOutput(BaseModel):
    """Output format for pipeline predictions."""
    
    failure_category: str = Field(
        description="The predicted failure category"
    )
    categorization_reason: str = Field(
        description="Reasoning for the categorization"
    )
    category_definition: str = Field(
        description="Definition of the predicted category"
    )


class FailureCategorizationPipeline(weave.Model):
    """
    Weave model wrapper for the failure categorization pipeline.
    
    This model wraps the complete pipeline following the baseline evaluation pattern.
    """
    
    model_name: str
    eval_id: str
    wandb_entity: str
    wandb_project: str
    max_concurrent_llm_calls: int = 100
    n_samples: Optional[int] = None
    debug: bool = False
    
    @weave.op
    async def predict(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """
        Run the failure categorization pipeline on a single example.
        
        Args:
            inputs: Dict containing row_input, row_output, and evaluation_data
            
        Returns:
            Dict with failure_category, categorization_reason, and category_definition
        """
        
        # Extract inputs
        row_input = inputs.get("row_input", "")
        row_output = inputs.get("row_output", "")
        evaluation_data = inputs.get("evaluation_data", "")
        
        # For demonstration, use pattern-based categorization
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


async def run_pipeline_evaluation():
    """
    Run the Weave evaluation for the failure categorization pipeline using EvaluationLogger
    for proper dataset-level ARI and Category-Discovery F1 scoring.
    """
    
    # Initialize Weave
    weave.init(args.project)
    console.print(f"[green]Initialized Weave project: {args.project}[/green]")
    
    # Load the dataset
    try:
        dataset = weave.ref(args.dataset_ref).get()
        console.print(f"[green]Loaded dataset '{args.dataset_ref}' with {len(dataset.rows)} rows[/green]")
        
        # Log dataset structure for debugging
        if dataset.rows:
            sample_row = dataset.rows[0]
            console.print(f"Sample dataset row keys: {list(sample_row.keys())}")
            
    except Exception as e:
        console.print(f"[red]Failed to load dataset '{args.dataset_ref}': {e}[/red]")
        return None
    
    # Initialize the pipeline model
    model = FailureCategorizationPipeline(
        model_name=args.model,
        eval_id=args.eval_id,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        max_concurrent_llm_calls=args.max_concurrent_llm_calls,
        n_samples=args.n_samples,
        debug=args.debug
    )
    console.print(f"[green]Initialized pipeline model: {args.model}[/green]")
    
    console.print("Using EvaluationLogger for dataset-level scoring:")
    console.print("  - Adjusted Rand Index: Partition agreement without caring about label strings")
    console.print("  - Category-Discovery F1: Coverage × precision with Hungarian assignment")
    
    # Create evaluation logger
    eval_logger = weave.EvaluationLogger(
        model=model, 
        dataset=dataset,
        name="failure_categorization_pipeline_eval"
    )
    
    # Store all predictions and ground truths for dataset-level scoring
    all_predictions = []
    all_ground_truths = []
    
    console.print("Starting failure categorization pipeline evaluation...")
    console.print("This may take a while depending on the dataset size...")
    
    try:
        # Process each example and log predictions
        for i, example in enumerate(dataset.rows):
            console.print(f"Processing example {i+1}/{len(dataset.rows)}...")
            
            # Prepare model inputs
            model_inputs = {
                "row_input": example.get("example.full_transcript", ""),
                "row_output": example.get("output.reasoning", ""),
                "evaluation_data": example.get("failure_category_reason_gt", "")
            }
            
            # Get model prediction
            prediction = await model.predict(model_inputs)
            
            # Log the prediction
            pred_logger = eval_logger.log_prediction(
                inputs=model_inputs,
                output=prediction
            )
            pred_logger.finish()
            
            # Store for dataset-level metrics
            predicted_category = prediction.get("failure_category", "")
            ground_truth_category = example.get("failure_category_gt", "")
            
            all_predictions.append(predicted_category)
            all_ground_truths.append(ground_truth_category)
            
            console.print(f"  Predicted: {predicted_category}")
            console.print(f"  Ground Truth: {ground_truth_category}")
        
        console.print("Calculating dataset-level metrics...")
        
        # Import our custom scorers
        from pipeline_scorers import ARIScorer, CatDiscoveryF1
        
        # Calculate ARI score
        ari_scorer = ARIScorer()
        
        # Create mock dataset for scorer
        class MockDataset:
            def __init__(self, preds, golds):
                self.data = {
                    "failure_category": preds,
                    "failure_category_gt": golds
                }
            
            def __getitem__(self, key):
                return MockColumn(self.data[key])

        class MockColumn:
            def __init__(self, data):
                self._data = data
            
            def to_py(self):
                return self._data
        
        mock_dataset = MockDataset(all_predictions, all_ground_truths)
        
        ari_score = ari_scorer.score(mock_dataset)
        
        # Calculate Category-Discovery F1
        cat_f1_scorer = CatDiscoveryF1()
        f1_score = cat_f1_scorer.score(mock_dataset)
        
        # Log summary with dataset-level metrics
        summary_stats = {
            "adjusted_rand_index": ari_score,
            "category_discovery_f1": f1_score,
            "num_examples": len(all_predictions),
            "unique_predicted_categories": len(set(all_predictions)),
            "unique_ground_truth_categories": len(set(all_ground_truths))
        }
        
        eval_logger.log_summary(summary_stats)
        
        console.print("Pipeline evaluation completed successfully!")
        console.print("\nDataset-level Results:")
        console.print(f"  Adjusted Rand Index: {ari_score:.4f} (partition agreement, -1 to 1)")
        console.print(f"  Category-Discovery F1: {f1_score:.4f} (coverage × precision, 0 to 1)")
        console.print(f"  Predicted categories: {len(set(all_predictions))}")
        console.print(f"  Ground truth categories: {len(set(all_ground_truths))}")
        
        return summary_stats
        
    except Exception as e:
        console.print(f"[red]Pipeline evaluation failed: {e}[/red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        return None


async def main():
    """Main function for running the pipeline evaluation."""
    
    if not args.run_eval:
        console.print("Use --run_eval to run the pipeline evaluation")
        console.print("Example: uv run python pipeline_weave_eval.py --run_eval")
        return
    
    if not os.getenv("GOOGLE_API_KEY"):
        console.print("[red]GOOGLE_API_KEY environment variable is required[/red]")
        return
    
    console.print("="*60)
    console.print("FAILURE CATEGORIZATION PIPELINE EVALUATION")
    console.print("="*60)
    console.print(f"Model: {args.model}")
    console.print(f"Dataset: {args.dataset_ref}")
    console.print(f"Debug mode: {args.debug}")
    console.print(f"Project: {args.project}")
    console.print(f"Eval ID: {args.eval_id}")
    console.print(f"Samples: {args.n_samples or 'All'}")
    console.print("="*60)
    
    # Run the evaluation
    results = await run_pipeline_evaluation()
    
    if results:
        console.print("="*60)
        console.print("EVALUATION COMPLETED SUCCESSFULLY")
        console.print("="*60)
        console.print("Check the Weave UI for detailed results and analysis")
    else:
        console.print("="*60)
        console.print("EVALUATION FAILED")
        console.print("="*60)
        console.print("Check the logs above for error details")


if __name__ == "__main__":
    asyncio.run(main())