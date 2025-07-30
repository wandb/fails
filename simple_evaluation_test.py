#!/usr/bin/env python3
"""
Simplified test of the corrected ARI and Category-Discovery F1 scorers.
This creates a mock model to test the scoring methodology.
"""

import asyncio
import weave
from typing import Dict
from dotenv import load_dotenv

load_dotenv()

# Mock model that returns simple categorizations for testing
class MockFailureCategorizationModel(weave.Model):
    """Mock model for testing the scorers."""
    
    @weave.op
    async def predict(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Return mock categorization based on simple patterns."""
        
        transcript = inputs.get("row_input", "").lower()
        
        # Simple pattern-based categorization for testing
        if "format" in transcript or "structure" in transcript:
            category = "format_error"
            reason = "Output appears to have formatting issues"
            definition = "Issues with the structure or format of the output"
        elif "logic" in transcript or "reasoning" in transcript:
            category = "logic_error"
            reason = "Output shows flawed logical reasoning"
            definition = "Errors in logical reasoning or inference"
        elif "halluc" in transcript or "false" in transcript:
            category = "hallucination"
            reason = "Output contains false or fabricated information"
            definition = "Model generated false or nonsensical information"
        else:
            category = "other"
            reason = "Does not fit standard categories"
            definition = "Miscellaneous failure category"
        
        return {
            "failure_category": category,
            "categorization_reason": reason,
            "category_definition": definition
        }


async def test_scorers_with_mock_data():
    """Test the corrected scorers with mock evaluation data."""
    
    # Initialize Weave
    weave.init("wandb-applied-ai-team/eval-failures")
    print("🔄 Initialized Weave project")
    
    # Load the actual dataset
    try:
        dataset = weave.ref("speaker_classification_failure_annotation:v0").get()
        print(f"✅ Loaded dataset with {len(dataset.rows)} rows")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    # Create mock model
    model = MockFailureCategorizationModel()
    print("✅ Created mock model")
    
    # Create evaluation logger
    eval_logger = weave.EvaluationLogger(
        model=model,
        dataset=dataset,
        name="mock_failure_categorization_test"
    )
    print("✅ Created evaluation logger")
    
    # Process examples and collect predictions
    all_predictions = []
    all_ground_truths = []
    
    print(f"\n🔄 Processing {len(dataset.rows)} examples...")
    
    for i, example in enumerate(dataset.rows[:5]):  # Test with first 5 examples
        print(f"  Example {i+1}/5...")
        
        # Prepare inputs
        model_inputs = {
            "row_input": example.get("example.full_transcript", ""),
            "row_output": example.get("output.reasoning", ""),
            "evaluation_data": example.get("failure_category_reason_gt", "")
        }
        
        # Get prediction
        prediction = await model.predict(model_inputs)
        
        # Log prediction
        pred_logger = eval_logger.log_prediction(
            inputs=model_inputs,
            output=prediction
        )
        pred_logger.finish()
        
        # Store for scoring
        predicted_category = prediction["failure_category"]
        ground_truth_category = example.get("failure_category_gt", "")
        
        all_predictions.append(predicted_category)
        all_ground_truths.append(ground_truth_category)
        
        print(f"    Predicted: {predicted_category}")
        print(f"    Ground Truth: {ground_truth_category}")
    
    print(f"\n🔄 Calculating dataset-level metrics...")
    
    # Test the corrected scorers
    from pipeline_scorers import ARIScorer, CatDiscoveryF1
    
    # Create mock dataset for scorers
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
    
    # Calculate ARI
    ari_scorer = ARIScorer()
    ari_score = ari_scorer.score(mock_dataset)
    
    # Calculate Category-Discovery F1
    cat_f1_scorer = CatDiscoveryF1()
    f1_score = cat_f1_scorer.score(mock_dataset)
    
    # Log summary
    summary_stats = {
        "adjusted_rand_index": ari_score,
        "category_discovery_f1": f1_score,
        "num_examples": len(all_predictions),
        "unique_predicted_categories": len(set(all_predictions)),
        "unique_ground_truth_categories": len(set(all_ground_truths)),
        "predictions": all_predictions,
        "ground_truths": all_ground_truths
    }
    
    eval_logger.log_summary(summary_stats)
    
    print(f"\n✅ EVALUATION COMPLETED!")
    print(f"📊 Results:")
    print(f"  Adjusted Rand Index: {ari_score:.4f} (partition agreement, -1 to 1)")
    print(f"  Category-Discovery F1: {f1_score:.4f} (coverage × precision, 0 to 1)")
    print(f"  Examples processed: {len(all_predictions)}")
    print(f"  Predicted categories: {set(all_predictions)}")
    print(f"  Ground truth categories: {set(all_ground_truths)}")
    
    return summary_stats


if __name__ == "__main__":
    result = asyncio.run(test_scorers_with_mock_data())