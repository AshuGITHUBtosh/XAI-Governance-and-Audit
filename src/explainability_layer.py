from pathlib import Path
import json
import sys
import re
import pandas as pd
import shap
import joblib
import numpy as np
from typing import Optional, Dict

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_training import preprocess_for_model


def _extract_tree_model(model):
    """
    Extract a SHAP-compatible tree model from wrappers like
    VotingClassifier or CalibratedClassifierCV.
    """
    from sklearn.ensemble import VotingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    import xgboost as xgb

    # Unwrap VotingClassifier first
    if isinstance(model, VotingClassifier):
        estimators = model.estimators_
    else:
        estimators = [model]

    for est in estimators:
        # Unwrap CalibratedClassifierCV  ✅ Fixed: handles calibrated RF
        if isinstance(est, CalibratedClassifierCV):
            inner = est.estimator if hasattr(est, 'estimator') else est.base_estimator
            est = inner

        # Prefer XGBoost (fastest SHAP)
        if isinstance(est, xgb.XGBClassifier):
            return est

    # Fallback: return first unwrapped estimator
    for est in estimators:
        if isinstance(est, CalibratedClassifierCV):
            return est.estimator if hasattr(est, 'estimator') else est.base_estimator
        return est

    return model


def explain_model(model, X: pd.DataFrame) -> Dict[str, float]:
    """
    Explain model using SHAP and return top 10 features by mean |SHAP|.
    Safely handles VotingClassifier and CalibratedClassifierCV wrappers.
    """
    tree_model = _extract_tree_model(model)

    try:
        explainer = shap.TreeExplainer(tree_model)
        shap_values = explainer.shap_values(X)
    except Exception as e:
        # Last resort: use KernelExplainer on a sample (slower but always works)
        print(f"TreeExplainer failed ({e}), falling back to KernelExplainer on 50 samples")
        sample = shap.sample(X, 50)
        explainer = shap.KernelExplainer(
            lambda x: tree_model.predict_proba(pd.DataFrame(x, columns=X.columns))[:, 1],
            sample
        )
        shap_values = explainer.shap_values(sample)
        X = sample  # align feature names

    # Handle both list (binary clf) and array outputs
    if isinstance(shap_values, list):
        mv = np.abs(shap_values[1]).mean(axis=0)
    else:
        mv = np.abs(shap_values).mean(axis=0)

    mv = np.asarray(mv).flatten()
    feat_names = list(X.columns)
    summary = {feat: float(mv[i]) for i, feat in enumerate(feat_names)}

    sorted_features = sorted(summary.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_features[:10])


def explain(model_path: str, data_path: str, out_path: str):
    """CLI entry point: load saved model + data, write SHAP summary JSON."""
    model = joblib.load(model_path)
    tree_model = _extract_tree_model(model)

    df = pd.read_parquet(data_path)
    target = _detect_target(df)
    if target is not None:
        X, _ = preprocess_for_model(df.copy(), target)
    else:
        X = df.copy()
        for col in X.select_dtypes(include=['number']).columns:
            X[col] = X[col].fillna(X[col].median())
        for col in X.select_dtypes(include=['object', 'category']).columns:
            if pd.api.types.is_categorical_dtype(X[col]):
                X[col] = X[col].cat.add_categories(['__missing__']).fillna('__missing__')
            else:
                X[col] = X[col].fillna('__missing__')
        X = pd.get_dummies(X, drop_first=True)
        X.columns = [re.sub(r"[^0-9a-zA-Z_]", "_", str(c)) for c in X.columns]

    # Align features to model's expected feature set
    try:
        model_features = tree_model.get_booster().feature_names
        for f in model_features:
            if f not in X.columns:
                X[f] = 0
        X = X[model_features]
    except Exception:
        pass  # RF models don't have get_booster — use X as-is

    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        mv = np.mean([np.abs(s).mean(axis=0) for s in shap_values], axis=0)
    else:
        mv = np.abs(shap_values).mean(axis=0)

    feat_names = list(X.columns)
    summary = {feat: float(mv[i]) for i, feat in enumerate(feat_names)}

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'mean_abs_shap': summary}, f, indent=2)

    print(f"Wrote explainability summary to {out}")


def _detect_target(df: pd.DataFrame) -> Optional[str]:
    for t in ['default', 'target', 'y', 'label', 'class', 'status']:
        if t in df.columns:
            return t
    return None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data',  type=str, default='data/processed/test.parquet')
    parser.add_argument('--out',   type=str, default='reports/explainability_summary.json')
    args = parser.parse_args()
    explain(args.model, args.data, args.out)