"""
Data Drift Detection Module

Monitors data stability and detects feature drift between reference and new data
using Kolmogorov-Smirnov (KS) statistical test.
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from typing import Dict, Any, Tuple


def detect_feature_drift(reference_df: pd.DataFrame, new_df: pd.DataFrame, p_threshold: float = 0.05) -> Dict[str, Any]:
    """
    Detect drift between reference (training) data and new data using KS test.
    
    The Kolmogorov-Smirnov test compares cumulative distributions of two samples.
    If p-value < p_threshold, we reject the null hypothesis that distributions are identical,
    indicating significant drift.
    
    Args:
        reference_df: Reference/training data (baseline)
        new_df: New data to check for drift
        p_threshold: Significance level for drift detection (default: 0.05)
    
    Returns:
        Dictionary with drift results for each numeric feature:
        {
            "feature_name": {
                "p_value": float,
                "drift_detected": bool,
                "ks_statistic": float
            },
            ...
        }
    """
    drift_results = {}
    
    numeric_cols = reference_df.select_dtypes(include=["number"]).columns
    
    for col in numeric_cols:
        if col in new_df.columns:
            # Extract and drop missing values
            ref_values = reference_df[col].dropna()
            new_values = new_df[col].dropna()
            
            # Only perform test if both sets have data
            if len(ref_values) > 0 and len(new_values) > 0:
                # KS test: compares cumulative distributions
                ks_stat, p_value = ks_2samp(ref_values, new_values)
                
                drift_results[col] = {
                    "p_value": float(p_value),
                    "drift_detected": bool(p_value < p_threshold),
                    "ks_statistic": float(ks_stat)
                }
    
    return drift_results


def summarize_drift(drift_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize drift detection results.
    
    Args:
        drift_results: Output from detect_feature_drift()
    
    Returns:
        Summary dictionary with key statistics
    """
    if not drift_results:
        return {
            "total_features_monitored": 0,
            "features_with_drift": 0,
            "drift_percentage": 0.0,
            "drifted_features": []
        }
    
    total = len(drift_results)
    drifted = [col for col, metrics in drift_results.items() if metrics["drift_detected"]]
    
    return {
        "total_features_monitored": total,
        "features_with_drift": len(drifted),
        "drift_percentage": float(len(drifted) / total * 100) if total > 0 else 0.0,
        "drifted_features": drifted
    }
