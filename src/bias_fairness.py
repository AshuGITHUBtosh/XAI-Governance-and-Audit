import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


def evaluate_fairness(y_true, y_pred, sensitive):
    """
    Evaluate fairness metrics: demographic parity and equalized odds.
    Handles edge cases where a sensitive group has only one class in test set.
    """
    # ✅ Reset indexes to avoid alignment issues
    y_true    = pd.Series(y_true).reset_index(drop=True)
    y_pred    = pd.Series(y_pred).reset_index(drop=True)
    sensitive = pd.Series(sensitive).reset_index(drop=True)

    # ✅ Auto-bin continuous numeric sensitive attributes (e.g. age)
    # Without binning, 53 unique age values → tiny groups → DP diff = 1.0
    if pd.api.types.is_numeric_dtype(sensitive) and sensitive.nunique() > 10:
        sensitive = pd.cut(
            sensitive,
            bins=[0, 25, 35, 50, 100],
            labels=['young', 'young_adult', 'middle_aged', 'senior']
        ).astype(str)

    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'sensitive': sensitive})

    # Demographic Parity: difference in positive prediction rates across groups
    grp_pos_rate = df.groupby('sensitive')['y_pred'].mean()
    dp_diff = float(grp_pos_rate.max() - grp_pos_rate.min()) if len(grp_pos_rate) > 1 else 0.0

    # Equalized Odds: max difference in TPR and FPR across groups
    groups = df['sensitive'].unique()
    tprs, fprs = [], []

    for g in groups:
        sub = df[df['sensitive'] == g]

        unique_true = sub['y_true'].nunique()

        if unique_true < 2 or len(sub) < 2:
            tprs.append(0.0)
            fprs.append(0.0)
            continue

        try:
            cm = confusion_matrix(sub['y_true'], sub['y_pred'], labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        except ValueError:
            tpr, fpr = 0.0, 0.0

        tprs.append(tpr)
        fprs.append(fpr)

    if len(tprs) > 1:
        eo_diff = float(max(
            max(tprs) - min(tprs),
            max(fprs) - min(fprs)
        ))
    else:
        eo_diff = 0.0

    return {
        'demographic_parity_difference': dp_diff,
        'equalized_odds_difference':     eo_diff,
    }