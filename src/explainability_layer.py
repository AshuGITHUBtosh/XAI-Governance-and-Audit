from pathlib import Path
import json
import pandas as pd
import shap
import joblib
from typing import Optional

from src.model_training import preprocess_for_model


def _detect_target(df: pd.DataFrame) -> Optional[str]:
    for t in ['default', 'target', 'y', 'label', 'class', 'status']:
        if t in df.columns:
            return t
    return None


def explain(model_path: str, data_path: str, out_path: str):
    model = joblib.load(model_path)

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
        import re
        X.columns = [re.sub(r"[^0-9a-zA-Z_]", "_", str(c)) for c in X.columns]

    try:
        model_features = model.get_booster().feature_names
    except Exception:
        model_features = None

    if model_features is not None:
        for f in model_features:
            if f not in X.columns:
                X[f] = 0
        X = X[model_features]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    import numpy as np
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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data', type=str, default='data/processed/test.parquet')
    parser.add_argument('--out', type=str, default='reports/explainability_summary.json')
    args = parser.parse_args()
    explain(args.model, args.data, args.out)
