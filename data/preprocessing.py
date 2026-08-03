#!/usr/bin/env python3
"""
Data Preprocessing Module

This module contains functions for data preprocessing and feature engineering.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def load_data(filepath: str) -> pd.DataFrame:
    """Load data from CSV file.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Loaded DataFrame
    """
    return pd.read_csv(filepath)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess data.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.fillna(df.mean(numeric_only=True))
    
    return df


def normalize_features(X: np.ndarray) -> np.ndarray:
    """Normalize features to zero mean and unit variance.
    
    Args:
        X: Feature array
        
    Returns:
        Normalized feature array
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / (std + 1e-8)
