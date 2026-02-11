"""Trajectory analysis utilities: PSI, rolling cohort metrics, and simple CLI.

Functions:
- population_stability_index(expected, actual, bins=10)
- compute_psi_for_df(df_ref, df_cur, features, bins=10)
- cohort_performance_over_time(preds_df, time_col, group_col, period='7D')
- performance_by_bucket(preds_df, bucket_col, y_true='y_true', y_pred='y_pred')

CLI example:
  python -m src.trajectory.analysis --preds reports/model_predictions.csv --mode psi
"""
from __future__ import annotations
import argparse
from typing import List, Dict, Callable, Optional

import numpy as np
import pandas as pd


def _safe_pct(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    if total == 0:
        return np.zeros_like(counts, dtype=float)
    pct = counts.astype(float) / total
    # avoid zeros for log stability
    pct = np.where(pct == 0, 1e-8, pct)
    return pct


def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Compute PSI between two series (numeric or categorical).

    For numeric: use histogram bins; for categorical: use union of categories.
    """
    if expected.dropna().empty or actual.dropna().empty:
        return float('nan')

    if pd.api.types.is_numeric_dtype(expected) and pd.api.types.is_numeric_dtype(actual):
        # numeric: use shared bin edges from expected
        try:
            counts_exp, edges = np.histogram(expected.dropna(), bins=bins)
            counts_act, _ = np.histogram(actual.dropna(), bins=edges)
        except Exception:
            # fallback to quantile bins
            edges = np.quantile(expected.dropna(), np.linspace(0, 1, bins + 1))
            counts_exp, _ = np.histogram(expected.dropna(), bins=edges)
            counts_act, _ = np.histogram(actual.dropna(), bins=edges)
        pct_exp = _safe_pct(counts_exp)
        pct_act = _safe_pct(counts_act)
    else:
        # categorical
        cats = pd.Index(expected.dropna().unique()).union(actual.dropna().unique())
        counts_exp = expected.dropna().value_counts().reindex(cats, fill_value=0).values
        counts_act = actual.dropna().value_counts().reindex(cats, fill_value=0).values
        pct_exp = _safe_pct(counts_exp)
        pct_act = _safe_pct(counts_act)

    psi = np.sum((pct_exp - pct_act) * np.log(pct_exp / pct_act))
    return float(psi)


def compute_psi_for_df(df_ref: pd.DataFrame, df_cur: pd.DataFrame, features: List[str], bins: int = 10) -> Dict[str, float]:
    results: Dict[str, float] = {}
    for f in features:
        if f not in df_ref.columns or f not in df_cur.columns:
            results[f] = float('nan')
            continue
        try:
            results[f] = population_stability_index(df_ref[f], df_cur[f], bins=bins)
        except Exception:
            results[f] = float('nan')
    return results


def performance_by_bucket(preds_df: pd.DataFrame, bucket_col: str, y_true: str = 'y_true', y_pred: str = 'y_pred') -> pd.DataFrame:
    """Compute simple accuracy and counts per bucket."""
    if bucket_col not in preds_df.columns:
        raise ValueError(f"Bucket column {bucket_col} not found in preds dataframe")
    if y_true not in preds_df.columns or y_pred not in preds_df.columns:
        raise ValueError("Prediction or truth columns not found in preds dataframe")

    def _acc(g: pd.DataFrame) -> float:
        return float((g[y_true] == g[y_pred]).mean()) if len(g) > 0 else float('nan')

    out = preds_df.groupby(bucket_col).apply(lambda g: pd.Series({'count': len(g), 'accuracy': _acc(g)}))
    return out.reset_index()


def cohort_performance_over_time(preds_df: pd.DataFrame, time_col: Optional[str], group_col: str, freq: str = '7D') -> pd.DataFrame:
    """Return grouped performance over time periods.

    preds_df: expected to contain `y_true` and `y_pred`.
    If `time_col` is None, uses the dataframe index.
    Returns a dataframe with columns: [period_start, group, count, accuracy]
    """
    df = preds_df.copy()
    if time_col and time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
    else:
        # require a datetime-like index for resampling; attempt to parse if index is string
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            # create a synthetic time by integer index
            df.index = pd.date_range('2000-01-01', periods=len(df), freq='D')

    def _acc_series(g: pd.DataFrame) -> float:
        return float((g['y_true'] == g['y_pred']).mean()) if len(g) > 0 else float('nan')

    records = []
    for period_start, window in df.resample(freq):
        group = window.groupby(group_col)
        for grp_name, grp_df in group:
            records.append({'period_start': period_start, 'group': grp_name, 'count': len(grp_df), 'accuracy': _acc_series(grp_df)})

    return pd.DataFrame.from_records(records)


def _main_cli():
    parser = argparse.ArgumentParser(description='Trajectory analysis utilities')
    parser.add_argument('--preds', help='Predictions CSV (with y_true,y_pred and optional timestamp)', required=True)
    parser.add_argument('--mode', choices=['psi', 'cohort', 'bucket'], default='psi')
    parser.add_argument('--ref', help='Reference CSV (for psi)', default='data/processed/train.parquet')
    parser.add_argument('--features', help='Comma-separated features for PSI')
    parser.add_argument('--time_col', help='Timestamp column for cohort analysis', default='timestamp')
    parser.add_argument('--group_col', help='Group column for cohort analysis (e.g., age_bucket)', default='age_bucket')
    args = parser.parse_args()

    preds_df = pd.read_csv(args.preds)

    if args.mode == 'psi':
        ref = pd.read_parquet(args.ref)
        if args.features:
            feats = [f.strip() for f in args.features.split(',')]
        else:
            # default to numeric columns intersection
            feats = [c for c in ref.select_dtypes(include=[np.number]).columns if c in preds_df.columns]
        psi = compute_psi_for_df(ref, preds_df, feats)
        print('PSI per feature:')
        for k, v in psi.items():
            print(f"{k}: {v:.6f}")
    elif args.mode == 'bucket':
        out = performance_by_bucket(preds_df, args.group_col)
        print(out.to_string(index=False))
    else:
        out = cohort_performance_over_time(preds_df, args.time_col, args.group_col)
        print(out.head().to_string(index=False))


if __name__ == '__main__':
    _main_cli()
