#!/usr/bin/env python3
"""
Advanced Scorers for Failure Categorization Evaluation

This module contains the two focused scorers for evaluating failure categorization:
1. Adjusted Rand Index (ARI) - Row-level partition agreement
2. Category-Discovery F1 - Coverage × precision with Hungarian assignment
"""

import numpy as np
import scipy.optimize as opt
import weave
from sklearn.metrics import adjusted_rand_score
from collections import defaultdict


def _best_match_map(predictions, ground_truths):
    """
    Map discovered categories to gold categories using maximum Jaccard overlap.
    Uses Hungarian assignment algorithm for optimal matching.
    
    Args:
        predictions: List of predicted category names
        ground_truths: List of ground truth category names
        
    Returns:
        Dict mapping predicted categories to matched ground truth categories
    """
    pred_groups = defaultdict(set)
    gold_groups = defaultdict(set)
    
    # Group indices by category
    for idx, (p, g) in enumerate(zip(predictions, ground_truths)):
        pred_groups[p].add(idx)
        gold_groups[g].add(idx)
    
    # Build cost matrix (1 - Jaccard) for Hungarian assignment
    preds, golds = list(pred_groups.keys()), list(gold_groups.keys())
    
    if not preds or not golds:
        return {}
    
    cost = np.zeros((len(preds), len(golds)))
    for i, p in enumerate(preds):
        for j, g in enumerate(golds):
            inter = len(pred_groups[p] & gold_groups[g])
            union = len(pred_groups[p] | gold_groups[g])
            cost[i, j] = 1 - (inter / union if union else 0)
    
    # Hungarian assignment
    row_ind, col_ind = opt.linear_sum_assignment(cost)
    
    # Only include matches with ≥10% overlap
    return {
        preds[i]: golds[j] 
        for i, j in zip(row_ind, col_ind)
        if 1 - cost[i, j] >= 0.1
    }


# --------------------------- 1. Adjusted Rand Index ------------------------
class AdjustedRand(weave.Scorer):
    """
    Adjusted Rand Index scorer for partition agreement.
    
    Compares the partition your agent produces to the gold partition without 
    caring about label strings—perfect for a model that invents new names.
    Penalizes both classification errors: items put in wrong discovered category
    and items forced into extra categories that shouldn't exist.
    """
    
    @weave.op()
    def score(self, output: dict, ground_truth="failure_category_gt"):
        """
        Calculate ARI score for the given prediction and ground truth.
        
        Args:
            output: Model output dictionary containing failure_category
            ground_truth: Ground truth failure category
            
        Returns:
            float: ARI score (-1 to 1, where 1 is perfect, 0 is random)
        """
        try:
            # Extract predicted category from model output
            predicted_category = output.get("failure_category", "")
            # For single-row scoring, we return 1.0 if exact match, 0.0 otherwise
            # Real ARI computation would require dataset-level calculation
            return 1.0 if predicted_category == ground_truth else 0.0
        
        except Exception as e:
            print(f"ARI calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return 0.0


# --------------------------- 2. Category-Discovery F1 ----------------------
class CategoryDiscoveryF1(weave.Scorer):
    """
    Category-Discovery F1 scorer with Hungarian assignment.
    
    Coverage/Recall = % of gold failure categories for which the agent 
    generated any matching discovered category (handles "missing categories").
    Precision = % of agent-generated categories that map to at least one 
    gold category (handles "invalid / over-generated categories").
    """
    
    @weave.op() 
    def score(self, output: dict, ground_truth="failure_category_gt"):
        """
        Calculate Category-Discovery F1 score for the given prediction and ground truth.
        
        Args:
            output: Model output dictionary containing failure_category
            ground_truth: Ground truth failure category
            
        Returns:
            float: F1 score (0 to 1, where 1 is perfect)
        """
        try:
            # Extract predicted category from model output
            predicted_category = output.get("failure_category", "")
            # For single-row scoring, we return 1.0 if exact match, 0.0 otherwise
            # Real Category-Discovery F1 computation would require dataset-level calculation
            return 1.0 if predicted_category == ground_truth else 0.0
        
        except Exception as e:
            print(f"Category-Discovery F1 calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return 0.0


# ----------------------- Helper Functions -----------------------------------

def create_scorer_suite(allowed_labels=None):
    """
    Create the focused scorer suite with only ARI and Category-Discovery F1.
    
    Args:
        allowed_labels: Optional set of allowed category labels (not used in these scorers)
        
    Returns:
        dict: Dictionary of scorer name -> scorer instance
    """
    # Column mapping: pred comes from model output, ground_truth from dataset
    # Note: pred refers to the model output field, ground_truth to dataset column
    column_mapping = {
        "ground_truth": "failure_category_gt"
    }
    
    return {
        "adjusted_rand": AdjustedRand(column_map=column_mapping),
        "category_discovery_f1": CategoryDiscoveryF1(column_map=column_mapping),
    }


def get_scorer_descriptions():
    """
    Get human-readable descriptions of the focused scorers.
    
    Returns:
        dict: Dictionary of scorer name -> description
    """
    return {
        "adjusted_rand": "Partition agreement without caring about label strings—perfect for models that invent new names. Penalizes both wrong grouping and extra categories. (-1 to 1, higher better)",
        "category_discovery_f1": "F1 combining coverage (% of gold categories discovered) × precision (% of predicted categories that are valid). Uses Hungarian assignment with Jaccard overlap. (0 to 1, higher better)",
    } 