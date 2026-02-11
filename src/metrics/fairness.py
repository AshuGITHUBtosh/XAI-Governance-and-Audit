from typing import Union
import pandas as pd
import json
from pathlib import Path


def demographic_parity_difference(df: pd.DataFrame, y_pred_col: str, sensitive_col: str) -> float:
    grp = df.groupby(sensitive_col)[y_pred_col].mean()
    if len(grp) < 2:
        return 0.0
    return float(grp.max() - grp.min())


def equalized_odds_difference(df: pd.DataFrame, y_true_col: str, y_pred_col: str, sensitive_col: str) -> float:
    groups = df[sensitive_col].unique()
    tprs = {}
    fprs = {}
    for g in groups:
        sub = df[df[sensitive_col] == g]
        tp = ((sub[y_true_col] == 1) & (sub[y_pred_col] == 1)).sum()
        fn = ((sub[y_true_col] == 1) & (sub[y_pred_col] == 0)).sum()
        fp = ((sub[y_true_col] == 0) & (sub[y_pred_col] == 1)).sum()
        tn = ((sub[y_true_col] == 0) & (sub[y_pred_col] == 0)).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tprs[g] = tpr
        fprs[g] = fpr
    tpr_diff = max(tprs.values()) - min(tprs.values()) if len(tprs) > 1 else 0.0
    fpr_diff = max(fprs.values()) - min(fprs.values()) if len(fprs) > 1 else 0.0
    return float(max(tpr_diff, fpr_diff))


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compute fairness metrics from predictions CSV')
    parser.add_argument('--preds', type=str, required=True, help='CSV with y_true,y_pred and sensitive column')
    parser.add_argument('--sensitive', type=str, required=True)
    parser.add_argument('--y_true', type=str, default='y_true')
    parser.add_argument('--y_pred', type=str, default='y_pred')
    parser.add_argument('--out', type=str, default='reports/fairness_summary.json')
    args = parser.parse_args()

    df = pd.read_csv(args.preds)
    dp = demographic_parity_difference(df, args.y_pred, args.sensitive)
    eo = equalized_odds_difference(df, args.y_true, args.y_pred, args.sensitive)
    out = {'demographic_parity_difference': dp, 'equalized_odds_difference': eo}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"Wrote fairness summary to {args.out}")


if __name__ == '__main__':
    main()
