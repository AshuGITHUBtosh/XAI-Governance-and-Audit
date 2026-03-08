from pathlib import Path
import json
import sys
import re
from typing import Optional, List, Tuple, Dict, Any

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, f1_score, precision_recall_curve
)
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV,
    StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
from src.utils.io import save_csv
from src.drift import detect_feature_drift, summarize_drift
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def detect_target(df: pd.DataFrame) -> Optional[str]:
    for c in ["default", "target", "y", "label", "class", "status"]:
        if c in df.columns:
            return c
    return None


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ---------------------------------------------------------------------------
# Preprocessing & Feature Engineering
# ---------------------------------------------------------------------------

def preprocess_for_model(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()

    # Drop id-like columns
    for id_col in ("Unnamed: 0", "id", "ID", "Id"):
        if id_col in df.columns:
            df = df.drop(columns=[id_col])

    y = df[target].copy()

    # Fixed: cat.codes assigns labels alphabetically which flips 0/1 when
    # data.py already encoded to 0.0/1.0 floats. Map explicitly instead.
    if y.dtype == "object" or pd.api.types.is_categorical_dtype(y):
        str_map = {"good": 0, "bad": 1, "yes": 1, "no": 0, "true": 1, "false": 0}
        y_str = y.astype(str).str.strip().str.lower()
        mapped = y_str.map(str_map)
        if mapped.isna().any():
            print(f"  Warning: unknown target labels {y_str[mapped.isna()].unique()} — using cat.codes")
            mapped = y.astype("category").cat.codes.astype(float)
        y = mapped
    y = y.astype(float)
    X = df.drop(columns=[target])

    # --- Feature engineering for credit datasets ---
    if "credit_amount" in X.columns and "duration" in X.columns:
        X["credit_per_duration"] = X["credit_amount"] / (X["duration"] + 1)
        X["credit_per_duration"].fillna(X["credit_per_duration"].median(), inplace=True)

    if "age" in X.columns and "credit_amount" in X.columns:
        X["credit_per_age"] = X["credit_amount"] / (X["age"] + 1)
        X["credit_per_age"].fillna(X["credit_per_age"].median(), inplace=True)

    if "age" in X.columns and "duration" in X.columns:
        X["years_to_repay_ratio"] = X["duration"] / (X["age"] + 1)
        X["years_to_repay_ratio"].fillna(X["years_to_repay_ratio"].median(), inplace=True)

    # Binning
    if "age" in X.columns:
        X["age_group"] = pd.cut(
            X["age"], bins=[0, 25, 35, 45, 55, 100],
            labels=["young", "young_adult", "middle", "senior", "elder"]
        ).astype(str)

    if "duration" in X.columns:
        X["duration_group"] = pd.cut(
            X["duration"], bins=[0, 12, 24, 36, 60, 1000],
            labels=["short", "medium", "long", "very_long", "extreme"]
        ).astype(str)

    if "credit_amount" in X.columns:
        X["credit_amount_group"] = pd.cut(
            X["credit_amount"], bins=[0, 1000, 2500, 5000, 10000, 100000],
            labels=["small", "medium", "large", "very_large", "huge"]
        ).astype(str)

    # Polynomial features
    for col in ["credit_amount", "age", "duration"]:
        if col in X.columns:
            X[f"{col}_squared"] = X[col] ** 2

    # Fill missing values
    for col in X.select_dtypes(include=["number"]).columns:
        X[col] = X[col].fillna(X[col].median())
    for col in X.select_dtypes(include=["object", "category"]).columns:
        if pd.api.types.is_categorical_dtype(X[col]):
            X[col] = X[col].cat.add_categories(["__missing__"]).fillna("__missing__")
        else:
            X[col] = X[col].fillna("__missing__")

    X = pd.get_dummies(X, drop_first=True)
    X.columns = [re.sub(r"[^0-9a-zA-Z_]", "_", str(c)) for c in X.columns]
    return X, y


# ---------------------------------------------------------------------------
# Threshold Helpers
# ---------------------------------------------------------------------------

def find_optimal_threshold(y_true, y_prob) -> Tuple[float, float]:
    """Find threshold that maximises F1-score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.where(
        (precision + recall) == 0, 0,
        2 * (precision * recall) / (precision + recall)
    )
    idx = f1_scores.argmax()
    threshold = thresholds[idx] if idx < len(thresholds) else 0.5
    return float(threshold), float(f1_scores[idx])


def find_fair_threshold(y_true, y_prob, sensitive, fairness_limit: float = 0.2) -> float:
    """Threshold that satisfies fairness constraint and maximises accuracy."""
    from src.bias_fairness import evaluate_fairness

    best_threshold = 0.5
    best_accuracy = 0.0

    for t in np.linspace(0.1, 0.9, 17):
        y_pred = (y_prob >= t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        fairness = evaluate_fairness(y_true, y_pred, sensitive)
        bias = abs(fairness.get("demographic_parity_difference", 1.0))
        if bias <= fairness_limit and acc > best_accuracy:
            best_accuracy = acc
            best_threshold = t

    return float(best_threshold)


def analyze_threshold_tradeoff(y_true, y_prob, sensitive) -> List[Dict]:
    """Performance and fairness metrics across a range of thresholds."""
    from src.bias_fairness import evaluate_fairness

    results = []
    for t in np.linspace(0.1, 0.9, 17):
        y_pred = (y_prob >= t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        fairness = evaluate_fairness(y_true, y_pred, sensitive)
        results.append({
            "threshold": float(t),
            "accuracy": float(acc),
            "demographic_parity": float(fairness.get("demographic_parity_difference", 0)),
            "equalized_odds": float(fairness.get("equalized_odds_difference", 0)),
        })
    return results


# ---------------------------------------------------------------------------
# XGBoost Tuning
# ---------------------------------------------------------------------------

def tune_and_train_xgb(X_train, y_train, random_state: int = 42) -> xgb.XGBClassifier:
    """
    Two-stage tuning:
      1. RandomizedSearchCV on a held-out split to find best hyper-parameters.
      2. Retrain best config on full training data with early stopping.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15,
        random_state=random_state, stratify=y_train
    )

    unique, counts = np.unique(y_train, return_counts=True)
    pos_weight = float(counts[0]) / float(counts[1]) if len(unique) == 2 else 1.0

    param_grid = {
        "max_depth":        [3, 4, 5, 6],
        "learning_rate":    [0.01, 0.05, 0.1],
        "n_estimators":     [300, 500],
        "min_child_weight": [1, 3, 5],
        "subsample":        [0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.8],
        "gamma":            [0, 0.1, 0.3],
        "reg_alpha":        [0, 0.1, 0.5],   # L1
        "reg_lambda":       [1, 1.5, 2.0],   # L2
    }

    # Clean estimator for CV (no early stopping inside CV)
    xgb_cv = xgb.XGBClassifier(
        scale_pos_weight=pos_weight,
        random_state=random_state,
        eval_metric="auc",
        tree_method="hist",
        use_label_encoder=False,
    )

    search = RandomizedSearchCV(
        xgb_cv,
        param_distributions=param_grid,
        n_iter=50,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        n_jobs=-1,
        random_state=random_state,
        verbose=0,
    )
    search.fit(X_tr, y_tr)

    # Rebuild best model with early stopping on the held-out val set
    best_params = {k: v for k, v in search.best_params_.items() if k != "n_estimators"}

    final_model = xgb.XGBClassifier(
        **best_params,
        n_estimators=1000,           # High ceiling; early stopping will cut it
        scale_pos_weight=pos_weight,
        random_state=random_state,
        eval_metric="auc",
        early_stopping_rounds=40,
        tree_method="hist",
        use_label_encoder=False,
    )
    final_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return final_model


# ---------------------------------------------------------------------------
# Model Training (with SMOTE + Feature Selection)
# ---------------------------------------------------------------------------

def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    use_smote: bool = True,
) -> Tuple[Any, Any, Any]:
    """
    Returns (rf_calibrated, xgb_model, selector).
    Feature selector is returned so the same columns can be applied to X_test.
    """
    # --- SMOTE ---
    if use_smote and len(np.unique(y_train)) == 2:
        smote = SMOTE(random_state=random_state, k_neighbors=5)
        X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    else:
        X_balanced, y_balanced = X_train.copy(), y_train.copy()

    # --- Feature selection on balanced data ---
    selector_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    selector_rf.fit(X_balanced, y_balanced)
    selector = SelectFromModel(selector_rf, threshold="median", prefit=True)

    selected_cols = X_balanced.columns[selector.get_support()].tolist()
    X_bal_sel = X_balanced[selected_cols]
    print(f"  Feature selection: {X_balanced.shape[1]} → {len(selected_cols)} features kept")

    # --- Random Forest (balanced_subsample is better for RF on imbalanced data) ---
    rf_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
        bootstrap=True,
        oob_score=True,
    )
    rf_model.fit(X_bal_sel, y_balanced)
    print(f"  RF OOB score: {rf_model.oob_score_:.3f}")

    # Calibrate RF probabilities (improves threshold tuning significantly)
    # cv="prefit" was deprecated in newer sklearn — use a small CV split instead
    import sklearn
    sklearn_version = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    if sklearn_version >= (1, 2):
        # Newer sklearn: use cross-val calibration on balanced data
        rf_calibrated = CalibratedClassifierCV(rf_model, cv=3, method="isotonic")
        rf_calibrated.fit(X_bal_sel, y_balanced)
    else:
        # Older sklearn: prefit mode works fine
        rf_calibrated = CalibratedClassifierCV(rf_model, cv="prefit", method="isotonic")
        rf_calibrated.fit(X_bal_sel, y_balanced)

    # --- XGBoost ---
    xgb_model = tune_and_train_xgb(X_bal_sel, y_balanced, random_state)

    return rf_calibrated, xgb_model, selector


# ---------------------------------------------------------------------------
# Main Public API  — used by Streamlit dashboard
# ---------------------------------------------------------------------------

def train_and_evaluate_df(
    df: pd.DataFrame,
    target: str,
    sensitive: str,
    test_size: float = 0.2,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Full pipeline: preprocess → feature-select → train → evaluate → fairness.
    Returns a result dict consumed by the Streamlit dashboard.
    """
    if sensitive not in df.columns:
        raise ValueError(f"Sensitive column '{sensitive}' not found in dataset")

    sensitive_series = df[sensitive]
    X, y = preprocess_for_model(df, target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y if len(y.unique()) > 1 else None,
    )

    # --- Quick cross-val AUC before heavy training (honest estimate) ---
    quick_rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced",
        random_state=seed, n_jobs=-1
    )
    cv_scores = cross_val_score(
        quick_rf, X_train, y_train,
        cv=StratifiedKFold(5, shuffle=True, random_state=seed),
        scoring="roc_auc",
    )
    print(f"Cross-val AUC (before final training): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # --- Drift detection (on full feature set before selection) ---
    drift_results = detect_feature_drift(X_train, X_test, p_threshold=0.05)
    drift_summary = summarize_drift(drift_results)

    # --- Train (returns selector too) ---
    rf_model, xgb_model, selector = train_models(X_train, y_train, seed)

    # Apply consistent feature selection to both splits
    selected_cols = X_train.columns[selector.get_support()].tolist()
    X_train_sel = X_train[selected_cols]
    X_test_sel  = X_test[selected_cols]

    # --- Ensemble on selected features ---
    # ✅ Fix: VotingClassifier doesn't pass eval_set, so XGBoost's early_stopping
    # raises "Must have at least 1 validation dataset". Clone XGB without it.
    xgb_for_ensemble = xgb.XGBClassifier(**{
        k: v for k, v in xgb_model.get_params().items()
        if k not in ("early_stopping_rounds", "callbacks")
    })
    xgb_for_ensemble.fit(X_train_sel, y_train)

    ensemble = VotingClassifier(
        estimators=[("rf", rf_model), ("xgb", xgb_for_ensemble)],
        voting="soft",
    )
    ensemble.fit(X_train_sel, y_train)

    models = {
        "RandomForest": rf_model,
        "XGBoost":      xgb_model,
        "Ensemble":     ensemble,
    }

    model_metrics = {}
    best_model_name = None
    best_model      = None
    best_roc_auc    = -1.0

    for name, model in models.items():
        y_pred = model.predict(X_test_sel)
        y_prob = model.predict_proba(X_test_sel)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }

        if y_prob is not None and len(np.unique(y_test)) == 2:
            roc_auc = float(roc_auc_score(y_test, y_prob))
            opt_thresh, max_f1 = find_optimal_threshold(y_test, y_prob)
            tuned_pred = (y_prob >= opt_thresh).astype(int)

            metrics.update({
                "roc_auc":           roc_auc,
                "optimal_threshold": opt_thresh,
                "tuned_accuracy":    float(accuracy_score(y_test, tuned_pred)),
                "tuned_classification_report": classification_report(
                    y_test, tuned_pred, output_dict=True
                ),
            })

            if roc_auc > best_roc_auc:
                best_roc_auc    = roc_auc
                best_model      = model
                best_model_name = name

        else:
            if best_model is None:
                best_model      = model
                best_model_name = name

        model_metrics[name] = metrics
        print(f"  {name}: acc={metrics['accuracy']:.3f}"
              + (f", AUC={metrics.get('roc_auc', 0):.3f}" if "roc_auc" in metrics else ""))

    print(f"Best model: {best_model_name} (AUC={best_roc_auc:.3f})")

    # --- Best model probability output ---
    y_prob_best = (
        best_model.predict_proba(X_test_sel)[:, 1]
        if hasattr(best_model, "predict_proba") else None
    )

    # --- Sensitive attribute for test rows ---
    sensitive_test = sensitive_series.loc[X_test.index]

    # --- Fairness-aware threshold ---
    fair_threshold = 0.5
    if y_prob_best is not None and len(np.unique(y_test)) == 2:
        fair_threshold = find_fair_threshold(y_test, y_prob_best, sensitive_test, fairness_limit=0.2)
    print(f"  Fair threshold selected: {fair_threshold:.2f}")

    # --- Final predictions ---
    y_pred_final = (
        (y_prob_best >= fair_threshold).astype(int)
        if y_prob_best is not None
        else best_model.predict(X_test_sel)
    )

    # --- Fairness evaluation ---
    from src.bias_fairness import evaluate_fairness
    fairness = evaluate_fairness(y_test, y_pred_final, sensitive_test)

    # --- Threshold tradeoff analysis ---
    threshold_analysis = []
    if y_prob_best is not None and len(np.unique(y_test)) == 2:
        threshold_analysis = analyze_threshold_tradeoff(y_test, y_prob_best, sensitive_test)

    return {
        "best_model":         best_model,
        "best_model_name":    best_model_name,
        "selector":           selector,
        "selected_cols":      selected_cols,
        "model_metrics":      model_metrics,
        "cv_auc_mean":        float(cv_scores.mean()),
        "cv_auc_std":         float(cv_scores.std()),
        "fairness":           fairness,
        "fair_threshold":     float(fair_threshold),
        "threshold_analysis": threshold_analysis,
        "drift":              drift_results,
        "drift_summary":      drift_summary,
        # Pass selected feature sets downstream (e.g. to SHAP)
        "X_train":            X_train_sel,
        "X_test":             X_test_sel,
        "y_train":            y_train,
        "y_test":             y_test,
        "y_pred":             y_pred_final,
        "y_prob":             y_prob_best,
    }


# ---------------------------------------------------------------------------
# CLI entry point (train/test parquet files)
# ---------------------------------------------------------------------------

def train_and_evaluate(
    train_path: str,
    test_path: str,
    target: str,
    model_out: str,
    metrics_out: str,
    seed: int = 42,
):
    train = pd.read_parquet(train_path)
    test  = pd.read_parquet(test_path)

    if target is None:
        raise ValueError("Target must be specified")

    X_train, y_train = preprocess_for_model(train, target)
    X_test,  y_test  = preprocess_for_model(test,  target)

    # Drop NaN targets
    X_train = X_train[y_train.notna()];  y_train = y_train[y_train.notna()]
    X_test  = X_test[y_test.notna()];   y_test  = y_test[y_test.notna()]

    # Align columns
    common_cols = sorted(set(X_train.columns) & set(X_test.columns))
    X_train, X_test = X_train[common_cols], X_test[common_cols]
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows | Features: {len(common_cols)}")

    # Train
    rf_model, xgb_model, selector = train_models(X_train, y_train, seed)

    selected_cols = X_train.columns[selector.get_support()].tolist()
    X_train_sel = X_train[selected_cols]
    X_test_sel  = X_test[selected_cols]

    # Fix: clone XGB without early_stopping_rounds for VotingClassifier
    xgb_for_ensemble = xgb.XGBClassifier(**{
        k: v for k, v in xgb_model.get_params().items()
        if k not in ("early_stopping_rounds", "callbacks")
    })
    xgb_for_ensemble.fit(X_train_sel, y_train)

    ensemble = VotingClassifier(
        estimators=[("rf", rf_model), ("xgb", xgb_for_ensemble)],
        voting="soft",
    )
    ensemble.fit(X_train_sel, y_train)

    models = {"RandomForest": rf_model, "XGBoost": xgb_model, "Ensemble": ensemble}
    model_metrics = {}
    best_model, best_model_name, best_roc_auc = None, None, -1.0

    for name, model in models.items():
        y_pred = model.predict(X_test_sel)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }
        if len(np.unique(y_test.dropna())) == 2:
            try:
                y_prob = model.predict_proba(X_test_sel)[:, 1]
                roc_auc = float(roc_auc_score(y_test, y_prob))
                opt_thresh, _ = find_optimal_threshold(y_test, y_prob)
                tuned_pred = (y_prob >= opt_thresh).astype(int)
                metrics.update({
                    "roc_auc":           roc_auc,
                    "optimal_threshold": float(opt_thresh),
                    "tuned_accuracy":    float(accuracy_score(y_test, tuned_pred)),
                })
                if roc_auc > best_roc_auc:
                    best_roc_auc    = roc_auc
                    best_model      = model
                    best_model_name = name
            except Exception as e:
                print(f"  Warning ({name}): {e}")
        if best_model is None:
            best_model, best_model_name = model, name

        model_metrics[name] = metrics
        print(f"  {name}: acc={metrics['accuracy']:.3f}"
              + (f", AUC={metrics.get('roc_auc', 0):.3f}" if 'roc_auc' in metrics else ""))

    # Save
    model_path = Path(model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, str(model_path.with_suffix(".pkl")))

    save_json({
        "best_model":    best_model_name,
        "best_roc_auc":  best_roc_auc,
        "model_metrics": model_metrics,
    }, Path(metrics_out))

    # Save predictions
    try:
        y_pred_final = best_model.predict(X_test_sel)
        preds_df = X_test_sel.copy()
        preds_df["y_true"] = list(y_test)
        preds_df["y_pred"] = list(y_pred_final)
        save_csv(preds_df.reset_index(drop=True), Path("reports/model_predictions.csv"))
        print("Saved predictions to reports/model_predictions.csv")
    except Exception as e:
        print(f"Warning: could not save predictions — {e}")

    print(f"Best model ({best_model_name}) saved to {model_path}")
    print(f"Metrics saved to {metrics_out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    ROOT = Path(__file__).resolve().parent.parent
    sys.path.append(str(ROOT))

    parser = argparse.ArgumentParser(description="Train and evaluate risk models")
    parser.add_argument("--train",     default="data/processed/train.parquet")
    parser.add_argument("--test",      default="data/processed/test.parquet")
    parser.add_argument("--target",    default=None)
    parser.add_argument("--sensitive", default=None, help="Comma-separated sensitive columns")
    parser.add_argument("--model",     default="artifacts/model_xgb.json")
    parser.add_argument("--metrics",   default="reports/model_metrics.json")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    train_path = Path(args.train)
    test_path  = Path(args.test)
    if not train_path.exists() or not test_path.exists():
        print("Train/test files not found. Run ingestion first.")
        sys.exit(2)

    train_df = pd.read_parquet(train_path)
    target   = args.target or detect_target(train_df)
    if target is None:
        print("No target detected. Pass --target to the script.")
        sys.exit(3)

    print(f"Target: {target}")
    if args.sensitive:
        print(f"Sensitive columns: {[s.strip() for s in args.sensitive.split(',')]}")

    train_and_evaluate(str(train_path), str(test_path), target, args.model, args.metrics, args.seed)


if __name__ == "__main__":
    main()