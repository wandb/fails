#!/usr/bin/env python3
"""
Pipeline-specific Scorers for Failure Categorization Evaluation

This module contains two focused scorers for evaluating the pipeline:
1. Adjusted Rand Index (ARI) - Row-level partition agreement
2. Category-Discovery F1 - Coverage × precision with Hungarian assignment
"""

import weave
import numpy as np
import scipy.optimize as opt
from sklearn.metrics import adjusted_rand_score
from typing import Dict, Any, List, Set
from collections import defaultdict


def _best_match_map(predictions: List[str], ground_truths: List[str]) -> Dict[str, str]:
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


class ARIScorer(weave.Scorer):
    """
    Adjusted Rand Index scorer for partition agreement.
    
    Compares the partition your agent produces to the gold partition without 
    caring about label strings—perfect for a model that invents new names.
    Penalizes both classification errors: items put in wrong discovered category
    and items forced into extra categories that shouldn't exist.
    """
    
    @weave.op()
    def score(self, dataset, pred="failure_category", gold="failure_category_gt") -> float:
        """
        Calculate ARI score across the entire dataset.
        
        Args:
            dataset: Dataset with predictions and ground truth
            pred: Column name for predictions
            gold: Column name for ground truth
            
        Returns:
            float: ARI score (-1 to 1, where 1 is perfect, 0 is random)
        """
        try:
            # Extract predictions and ground truth from dataset
            if hasattr(dataset, 'to_py'):
                pred_values = dataset[pred].to_py()
                gold_values = dataset[gold].to_py()
            elif hasattr(dataset, '__getitem__') and pred in dataset.data:
                # Handle MockDataset format
                pred_values = dataset[pred].to_py()
                gold_values = dataset[gold].to_py()
            else:
                # Handle different dataset formats - assume it's iterable rows
                pred_values = [row.get(pred, "") for row in dataset]
                gold_values = [row.get(gold, "") for row in dataset]
            
            return adjusted_rand_score(gold_values, pred_values)
        
        except Exception as e:
            print(f"ARI calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return 0.0


class CatDiscoveryF1(weave.Scorer):
    """
    Category-Discovery F1 scorer with Hungarian assignment.
    
    Coverage/Recall = % of gold failure categories for which the agent 
    generated any matching discovered category (handles "missing categories").
    Precision = % of agent-generated categories that map to at least one 
    gold category (handles "invalid / over-generated categories").
    """
    
    @weave.op() 
    def score(self, dataset, pred="failure_category", gold="failure_category_gt") -> float:
        """
        Calculate Category-Discovery F1 score.
        
        Args:
            dataset: Dataset with predictions and ground truth
            pred: Column name for predictions  
            gold: Column name for ground truth
            
        Returns:
            float: F1 score (0 to 1, where 1 is perfect)
        """
        try:
            # Extract predictions and ground truth from dataset
            if hasattr(dataset, 'to_py'):
                pred_values = dataset[pred].to_py()
                gold_values = dataset[gold].to_py()
            elif hasattr(dataset, '__getitem__') and hasattr(dataset, 'data') and pred in dataset.data:
                # Handle MockDataset format
                pred_values = dataset[pred].to_py()
                gold_values = dataset[gold].to_py()
            else:
                # Handle different dataset formats - assume it's iterable rows
                pred_values = [row.get(pred, "") for row in dataset]
                gold_values = [row.get(gold, "") for row in dataset]
            
            # Get optimal mapping using Hungarian assignment
            mapping = _best_match_map(pred_values, gold_values)
            
            # Calculate metrics
            unique_preds = set(pred_values)
            unique_golds = set(gold_values)
            
            # Precision: % of predicted categories that have valid mappings
            mapped_preds = set(mapping.keys())
            precision = len(mapped_preds) / len(unique_preds) if unique_preds else 0.0
            
            # Coverage/Recall: % of gold categories that were discovered
            mapped_golds = set(mapping.values())  
            coverage = len(mapped_golds) / len(unique_golds) if unique_golds else 0.0
            
            # F1: harmonic mean
            f1 = 2 * precision * coverage / (precision + coverage + 1e-12)
            
            return f1
        
        except Exception as e:
            print(f"Category-Discovery F1 calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return 0.0


def create_pipeline_scorer_suite():
    """
    Create the focused scorer suite with only ARI and Category-Discovery F1.
    
    Returns:
        dict: Dictionary of scorer name -> scorer instance
    """
    # Column mapping for ground truth data
    column_mapping = {"ground_truth": "failure_category_gt"}
    
    return {
        "adjusted_rand_index": ARIScorer(column_map=column_mapping),
        "category_discovery_f1": CatDiscoveryF1(column_map=column_mapping),
    }


def get_pipeline_scorer_descriptions():
    """
    Get human-readable descriptions of the pipeline scorers.
    
    Returns:
        dict: Dictionary of scorer name -> description
    """
    return {
        "adjusted_rand_index": "Partition agreement without caring about label strings—perfect for models that invent new names. Penalizes both wrong grouping and extra categories. (-1 to 1, higher better)",
        "category_discovery_f1": "F1 combining coverage (% of gold categories discovered) × precision (% of predicted categories that are valid). Uses Hungarian assignment with Jaccard overlap. (0 to 1, higher better)",
    }