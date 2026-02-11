from pathlib import Path
import json
import sys
from typing import Optional, List

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib
from src.utils.io import save_csv


def detect_target(df: pd.DataFrame) -> Optional[str]:
    for c in ["default", "target", "y", "label", "class", "status"]:
        if c in df.columns:
            return c
    return None


def preprocess_for_model(df: pd.DataFrame, target: str) -> (pd.DataFrame, pd.Series):
    df = df.copy()
    for id_col in ("Unnamed: 0", "id", "ID", "Id"):
        if id_col in df.columns:
            df = df.drop(columns=[id_col])

    y = df[target].astype(float)
    X = df.drop(columns=[target])

    for col in X.select_dtypes(include=["number"]).columns:
        X[col] = X[col].fillna(X[col].median())
    for col in X.select_dtypes(include=["object", "category"]).columns:
        if pd.api.types.is_categorical_dtype(X[col]):
            X[col] = X[col].cat.add_categories(["__missing__"]).fillna("__missing__")
        else:
            X[col] = X[col].fillna("__missing__")

    X = pd.get_dummies(X, drop_first=True)
    import re
    X.columns = [re.sub(r"[^0-9a-zA-Z_]", "_", str(c)) for c in X.columns]
    return X, y


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def train_and_evaluate(train_path: str, test_path: str, target: str, model_out: str, metrics_out: str, seed: int = 42):
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    if target is None:
        raise ValueError("Target must be specified")

    X_train, y_train = preprocess_for_model(train, target)
    X_test, y_test = preprocess_for_model(test, target)

    model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=seed)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }
    if len(set(y_test.dropna())) == 2:
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
        except Exception:
            pass

    model_path = Path(model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(str(model_path))

    pkl_path = model_path.with_suffix('.pkl')
    joblib.dump(model, str(pkl_path))

    save_json(metrics, Path(metrics_out))

    preds_path = Path('reports/model_predictions.csv')
    try:
        preds_df = X_test.copy()
        preds_df['y_true'] = list(y_test)
        preds_df['y_pred'] = list(y_pred)
        preds_df = preds_df.reset_index(drop=True)
        save_csv(preds_df, preds_path)
        print(f"Saved predictions to {preds_path}")
    except Exception:
        pass

    print(f"Model saved to {model_path} and {pkl_path}")
    print(f"Metrics saved to {metrics_out}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train baseline XGBoost model")
    parser.add_argument("--train", type=str, default="data/processed/train.parquet")
    parser.add_argument("--test", type=str, default="data/processed/test.parquet")
    parser.add_argument("--target", type=str, default=None, help="Target column name")
    parser.add_argument("--sensitive", type=str, default=None, help="Comma-separated sensitive columns (optional)")
    parser.add_argument("--model", type=str, default="artifacts/model_xgb.json")
    parser.add_argument("--metrics", type=str, default="reports/model_metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_path = Path(args.train)
    test_path = Path(args.test)
    if not train_path.exists() or not test_path.exists():
        print("Train/test files not found. Run ingestion first.")
        sys.exit(2)

    train_df = pd.read_parquet(train_path)

    target = args.target or detect_target(train_df)
    if target is None:
        print("No target detected. Please pass --target to the script.")
        sys.exit(3)

    print(f"Using target: {target}")
    if args.sensitive:
        sens = [s.strip() for s in args.sensitive.split(',') if s.strip()]
        print(f"Sensitive columns: {sens}")

    train_and_evaluate(str(train_path), str(test_path), target, args.model, args.metrics, seed=args.seed)


if __name__ == "__main__":
    main()
