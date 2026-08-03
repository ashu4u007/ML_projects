#!/usr/bin/env python3
"""
Metrics Module

This module contains functions for calculating various evaluation metrics.
"""

import numpy as np
from typing import Tuple


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy score (0-1)
    """
    return np.mean(y_true == y_pred)


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate precision score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Precision score (0-1)
    """
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return tp / (tp + fp) if (tp + fp) > 0 else 0


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate recall score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Recall score (0-1)
    """
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate F1 score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        F1 score (0-1)
    """
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
