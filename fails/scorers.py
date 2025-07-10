#!/usr/bin/env python3
"""
Advanced Scorers for Failure Categorization Evaluation

This module contains sophisticated scoring functions for evaluating failure categorization
performance, including clustering quality metrics and embedding-based analysis.
"""

import math
import collections
import numpy as np
import weave
from typing import Optional, Set
from sklearn.metrics import (
    f1_score,
    adjusted_rand_score,
    silhouette_score,
    davies_bouldin_score,
)
from sentence_transformers import SentenceTransformer

# -------- Helper: sentence-level embeddings for free-text reasons ----------
_emb_model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-d vectors


def _embed(text_series):
    """Generate embeddings for a series of text strings."""
    return _emb_model.encode(text_series.to_py(), normalize_embeddings=True)


# --------------------------- 1. Exact micro-F1 ------------------------------
class MicroF1(weave.Scorer):
    """Exact match between predicted and gold failure labels."""
    
    @weave.op()
    def score(self, output, ground_truth):
        """
        Calculate micro-averaged F1 score for exact label matching.
        
        Args:
            output: Model prediction with failure_category
            ground_truth: Ground truth failure category
            
        Returns:
            dict: Score results with exact_match key
        """
        predicted = output.get("failure_category", "")
        exact_match = 1.0 if predicted == ground_truth else 0.0
        return {"exact_match": exact_match}


# --------------------------- 2. Adjusted Rand -------------------------------
class AdjustedRand(weave.Scorer):
    """Chance-corrected agreement between two clusterings."""
    
    @weave.op()
    def score(self, output, ground_truth):
        """
        Calculate category agreement score for individual examples.
        
        Args:
            output: Model prediction with failure_category
            ground_truth: Ground truth failure category
            
        Returns:
            dict: Score results with category_agreement key
        """
        predicted = output.get("failure_category", "")
        # For individual examples, we can measure categorical agreement
        agreement = 1.0 if predicted == ground_truth else 0.0
        return {"category_agreement": agreement}


# ------------ 3. Silhouette (cohesion / separation, cosine) -----------------
class Silhouette(weave.Scorer):
    """Reasoning quality analysis using embeddings."""
    
    @weave.op()
    def score(self, output, ground_truth):
        """
        Calculate reasoning quality score for individual examples.
        
        Args:
            output: Model prediction with categorization_reason
            ground_truth: Ground truth failure category
            
        Returns:
            dict: Score results with reasoning_quality key
        """
        reasoning = output.get("categorization_reason", "")
        # Simple reasoning quality metric: length and presence of key terms
        reasoning_length = len(reasoning.split())
        has_explanation = any(word in reasoning.lower() for word in ['because', 'due to', 'since', 'as', 'reason'])
        
        # Normalize reasoning quality score (0-1)
        length_score = min(reasoning_length / 20.0, 1.0)  # Cap at 20 words
        explanation_score = 1.0 if has_explanation else 0.0
        reasoning_quality = (length_score + explanation_score) / 2.0
        
        return {"reasoning_quality": reasoning_quality}


# -------------- 4. Davies–Bouldin Index (lower = better) --------------------
class DaviesBouldin(weave.Scorer):
    """Category confidence and definition quality scorer."""
    
    @weave.op()
    def score(self, output, ground_truth):
        """
        Calculate category confidence and definition quality.
        
        Args:
            output: Model prediction with categorization_reason and category_definition
            ground_truth: Ground truth failure category
            
        Returns:
            dict: Score results with definition_quality key
        """
        definition = output.get("category_definition", "")
        
        # Simple definition quality metric: length and structure
        def_length = len(definition.split())
        has_structure = any(word in definition.lower() for word in ['when', 'where', 'occurs', 'happens', 'refers'])
        
        # Normalize definition quality score (0-1)
        length_score = min(def_length / 15.0, 1.0)  # Cap at 15 words
        structure_score = 1.0 if has_structure else 0.0
        definition_quality = (length_score + structure_score) / 2.0
        
        return {"definition_quality": definition_quality}


# -------------------- 5. Cluster-balance entropy ----------------------------
class ClusterEntropy(weave.Scorer):
    """Measures prediction confidence and consistency."""
    
    @weave.op()
    def score(self, output, ground_truth):
        """
        Calculate prediction confidence based on reasoning consistency.
        
        Args:
            output: Model prediction with failure_category and categorization_reason
            ground_truth: Ground truth failure category
            
        Returns:
            dict: Score results with prediction_confidence key
        """
        predicted = output.get("failure_category", "")
        reasoning = output.get("categorization_reason", "")
        
        # Confidence based on prediction and reasoning alignment
        category_mentioned = predicted.lower() in reasoning.lower() if predicted and reasoning else False
        reasoning_length = len(reasoning.split()) if reasoning else 0
        
        # Simple confidence metric
        mention_score = 1.0 if category_mentioned else 0.0
        length_confidence = min(reasoning_length / 25.0, 1.0)  # Cap at 25 words
        prediction_confidence = (mention_score + length_confidence) / 2.0
        
        return {"prediction_confidence": prediction_confidence}


# -------------------- 6. Invalid-label rate ---------------------------------
class InvalidLabelRate(weave.Scorer):
    """Rate of predictions that fall outside allowed/expected labels."""
    
    def __init__(self, allowed_labels=None, **kwargs):
        """
        Initialize with allowed labels.
        
        Args:
            allowed_labels: Set of valid labels. If None, uses unique GT labels from dataset.
            **kwargs: Additional arguments passed to parent class (e.g., column_map)
        """
        super().__init__(**kwargs)
        self._allowed_labels = set(allowed_labels) if allowed_labels else None
    
    @weave.op()
    def score(self, output, ground_truth):
        """
        Calculate rate of invalid predictions.
        
        Args:
            output: Model prediction with failure_category
            ground_truth: Ground truth failure category
            
        Returns:
            dict: Score results with invalid_label key
        """
        predicted = output.get("failure_category", "")
        
        if self._allowed_labels is None:
            # If no predefined labels, accept any prediction as valid
            invalid_label = 0.0
        else:
            # Return 1.0 if invalid, 0.0 if valid
            invalid_label = 1.0 if predicted not in self._allowed_labels else 0.0
        
        return {"invalid_label": invalid_label}


# ----------------------- Helper Functions -----------------------------------

def create_scorer_suite(allowed_labels=None):
    """
    Create a complete suite of scorers for failure categorization evaluation.
    
    Args:
        allowed_labels: Optional set of allowed category labels
        
    Returns:
        dict: Dictionary of scorer name -> scorer instance
    """
    # Create scorers with column_map to map dataset columns to scorer arguments
    # Following the pattern: {scorer_argument: dataset_column_name}
    column_mapping = {"ground_truth": "failure_category_gt"}
    
    return {
        "micro_f1": MicroF1(column_map=column_mapping),
        "adjusted_rand": AdjustedRand(column_map=column_mapping),
        "silhouette": Silhouette(column_map=column_mapping),
        "davies_bouldin": DaviesBouldin(column_map=column_mapping),
        "cluster_entropy": ClusterEntropy(column_map=column_mapping),
        "invalid_rate": InvalidLabelRate(allowed_labels, column_map=column_mapping),
    }


def get_scorer_descriptions():
    """
    Get human-readable descriptions of all scorers.
    
    Returns:
        dict: Dictionary of scorer name -> description
    """
    return {
        "micro_f1": "Exact match between predicted and ground truth categories (0-1, higher better)",
        "adjusted_rand": "Category agreement between predicted and ground truth (0-1, higher better)", 
        "silhouette": "Reasoning quality based on length and explanation words (0-1, higher better)",
        "davies_bouldin": "Definition quality based on length and structure (0-1, higher better)",
        "cluster_entropy": "Prediction confidence based on reasoning consistency (0-1, higher better)",
        "invalid_rate": "Rate of predictions outside allowed labels (0-1, lower better)",
    } 