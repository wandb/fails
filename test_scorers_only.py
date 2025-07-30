#!/usr/bin/env python3
"""
Direct test of ARI and Category-Discovery F1 scorers without Weave complexity.
"""

from pipeline_scorers import ARIScorer, CatDiscoveryF1, _best_match_map

# Test data mimicking the real dataset structure
print("=" * 60)
print("TESTING CORRECTED SCORERS")
print("=" * 60)

# Mock dataset with realistic predictions vs ground truth
predictions = [
    "format_error",        # Should map to "formatting_issue" 
    "logic_error",         # Should map to "reasoning_flaw"
    "format_error",        # Should map to "formatting_issue"
    "hallucination",       # Should map to "factual_error"
    "reasoning_issue",     # Should map to "reasoning_flaw"
    "other",               # May not map to anything
    "format_error"         # Should map to "formatting_issue"
]

ground_truths = [
    "formatting_issue",    # Different name but same concept
    "reasoning_flaw",      # Different name but same concept  
    "formatting_issue",    # Perfect match in concept
    "factual_error",       # Different name but similar concept
    "reasoning_flaw",      # Different name but same concept
    "edge_case",           # Unique category not predicted
    "formatting_issue"     # Perfect match in concept
]

print(f"Predictions:   {predictions}")
print(f"Ground Truths: {ground_truths}")
print()

# Test Hungarian assignment mapping
print("🔍 Testing Hungarian Assignment Mapping:")
mapping = _best_match_map(predictions, ground_truths)
print(f"Discovered → Gold Mapping: {mapping}")
print()

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

dataset = MockDataset(predictions, ground_truths)

# Test ARI Scorer
print("📊 Testing ARI Scorer:")
ari_scorer = ARIScorer()
ari_score = ari_scorer.score(dataset)
print(f"ARI Score: {ari_score:.4f}")
print(f"Range: -1 to 1 (1=perfect partition agreement, 0=random)")
print()

# Test Category-Discovery F1
print("📊 Testing Category-Discovery F1:")
cat_f1_scorer = CatDiscoveryF1()
f1_score = cat_f1_scorer.score(dataset)
print(f"Category-Discovery F1: {f1_score:.4f}")
print(f"Range: 0 to 1 (1=perfect coverage and precision)")
print()

# Show detailed breakdown
unique_preds = set(predictions)
unique_golds = set(ground_truths)
mapped_preds = set(mapping.keys())
mapped_golds = set(mapping.values())

print("📋 Detailed Breakdown:")
print(f"  Unique Predicted Categories: {len(unique_preds)} → {unique_preds}")
print(f"  Unique Ground Truth Categories: {len(unique_golds)} → {unique_golds}")
print(f"  Successfully Mapped Predictions: {len(mapped_preds)} → {mapped_preds}")
print(f"  Ground Truth Categories Found: {len(mapped_golds)} → {mapped_golds}")
print()

coverage = len(mapped_golds) / len(unique_golds) if unique_golds else 0.0
precision = len(mapped_preds) / len(unique_preds) if unique_preds else 0.0

print(f"  Coverage (% of GT categories discovered): {coverage:.4f}")
print(f"  Precision (% of pred categories valid): {precision:.4f}")
print(f"  F1 (harmonic mean): {f1_score:.4f}")
print()

# Test with perfect predictions  
print("🎯 Testing with Perfect Predictions:")
perfect_preds = ground_truths.copy()  # Same as ground truth
perfect_dataset = MockDataset(perfect_preds, ground_truths)

perfect_ari = ari_scorer.score(perfect_dataset)
perfect_f1 = cat_f1_scorer.score(perfect_dataset)

print(f"Perfect ARI Score: {perfect_ari:.4f}")
print(f"Perfect F1 Score: {perfect_f1:.4f}")
print()

print("=" * 60)
print("✅ SCORER VERIFICATION COMPLETE")
print("=" * 60)
print()
print("Key Insights:")
print("- ARI measures partition agreement regardless of label names")
print("- Category-Discovery F1 uses Hungarian assignment to match similar categories")
print("- Both metrics handle the case where models invent new category names")
print("- Perfect scores show the metrics can detect ideal performance")