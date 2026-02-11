"""Runner to compute trajectory analysis and save a summary JSON.

Saves results to `reports/trajectory_summary.json`.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from src.trajectory.analysis import compute_psi_for_df, performance_by_bucket, cohort_performance_over_time


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / 'reports'
REPORTS.mkdir(parents=True, exist_ok=True)


def _to_serializable(obj: Any):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def run(save_path: str | Path = None) -> Dict[str, Any]:
    save_path = Path(save_path) if save_path else REPORTS / 'trajectory_summary.json'
    train_path = ROOT / 'data' / 'processed' / 'train.parquet'
    preds_path = ROOT / 'reports' / 'model_predictions.csv'

    result: Dict[str, Any] = {'meta': {}}

    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {preds_path}")
    preds = pd.read_csv(preds_path)
    result['meta']['preds_rows'] = int(len(preds))

    if train_path.exists():
        train = pd.read_parquet(train_path)
        result['meta']['train_rows'] = int(len(train))
        # choose numeric features present in both
        num_feats = [c for c in train.select_dtypes(include=[np.number]).columns if c in preds.columns]
        psi = compute_psi_for_df(train, preds, num_feats)
        # ensure plain floats
        result['psi'] = {k: float(v) if (v is not None and not (isinstance(v, float) and np.isnan(v))) else None for k, v in psi.items()}
    else:
        result['meta']['train_rows'] = 0
        result['psi'] = {}

    # bucket performance (age_bucket)
    if 'age_bucket' in preds.columns:
        perf_bucket = performance_by_bucket(preds, 'age_bucket')
        result['bucket_performance'] = perf_bucket.to_dict(orient='records')
    else:
        result['bucket_performance'] = []

    # cohort over time if possible
    time_col = 'timestamp' if 'timestamp' in preds.columns else None
    group_col = 'age_bucket' if 'age_bucket' in preds.columns else (preds.columns[0] if len(preds.columns) > 0 else None)
    if group_col:
        cohort_df = cohort_performance_over_time(preds, time_col, group_col)
        # convert datetimes to isoformat
        if not cohort_df.empty:
            cohort_df['period_start'] = cohort_df['period_start'].astype(str)
        result['cohort_over_time'] = cohort_df.to_dict(orient='records')
    else:
        result['cohort_over_time'] = []

    # save
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, default=_to_serializable, indent=2)

    return result


if __name__ == '__main__':
    out = run()
    print('Saved trajectory summary to reports/trajectory_summary.json')
