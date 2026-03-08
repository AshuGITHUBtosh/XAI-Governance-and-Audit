"""Runner to compute trajectory analysis and save a summary JSON.

Saves results to `reports/trajectory_summary.json`.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

# Resolve project root robustly regardless of working directory
ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / 'reports'
REPORTS.mkdir(parents=True, exist_ok=True)

from src.trajectory.analysis import compute_psi_for_df, performance_by_bucket, cohort_performance_over_time


def _to_serializable(obj: Any):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def run(save_path: str | Path = None, sensitive_col: str = None) -> Dict[str, Any]:
    """
    Args:
        save_path:     Where to write trajectory_summary.json
        sensitive_col: The sensitive attribute column name used in predictions.
                       ✅ Fixed: was hardcoded to 'age_bucket', now passed in.
    """
    save_path = Path(save_path) if save_path else REPORTS / 'trajectory_summary.json'
    train_path = ROOT / 'data' / 'processed' / 'train.parquet'
    preds_path = ROOT / 'reports' / 'model_predictions.csv'

    result: Dict[str, Any] = {'meta': {}}

    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {preds_path}")

    preds = pd.read_csv(preds_path)
    result['meta']['preds_rows'] = int(len(preds))

    # --- PSI against training data ---
    if train_path.exists():
        train = pd.read_parquet(train_path)
        result['meta']['train_rows'] = int(len(train))
        num_feats = [c for c in train.select_dtypes(include=[np.number]).columns if c in preds.columns]
        psi = compute_psi_for_df(train, preds, num_feats)
        result['psi'] = {
            k: float(v) if (v is not None and not (isinstance(v, float) and np.isnan(v))) else None
            for k, v in psi.items()
        }
    else:
        result['meta']['train_rows'] = 0
        result['psi'] = {}

    # --- Bucket performance ---
    # ✅ Fixed: use sensitive_col passed in, fall back gracefully
    bucket_col = None
    if sensitive_col and sensitive_col in preds.columns:
        bucket_col = sensitive_col
    elif 'sensitive' in preds.columns:
        bucket_col = 'sensitive'
    elif 'age_bucket' in preds.columns:
        bucket_col = 'age_bucket'

    if bucket_col:
        try:
            perf_bucket = performance_by_bucket(preds, bucket_col)
            result['bucket_performance'] = perf_bucket.to_dict(orient='records')
        except Exception as e:
            print(f"Warning: bucket performance failed — {e}")
            result['bucket_performance'] = []
    else:
        result['bucket_performance'] = []

    # --- Cohort over time ---
    # ✅ Fixed: only use a meaningful group_col, not a random feature column
    time_col  = 'timestamp' if 'timestamp' in preds.columns else None
    group_col = bucket_col  # reuse the same validated sensitive column

    if group_col:
        try:
            cohort_df = cohort_performance_over_time(preds, time_col, group_col)
            if not cohort_df.empty:
                cohort_df['period_start'] = cohort_df['period_start'].astype(str)
            result['cohort_over_time'] = cohort_df.to_dict(orient='records')
        except Exception as e:
            print(f"Warning: cohort analysis failed — {e}")
            result['cohort_over_time'] = []
    else:
        result['cohort_over_time'] = []

    # Save
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, default=_to_serializable, indent=2)

    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensitive', type=str, default=None,
                        help='Sensitive column name used in predictions CSV')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()
    out = run(save_path=args.out, sensitive_col=args.sensitive)
    print('Saved trajectory summary to reports/trajectory_summary.json')