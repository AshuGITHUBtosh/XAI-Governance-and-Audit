from pathlib import Path
import json
import sys
from typing import Optional, List, Tuple, Dict, Any

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
from src.utils.io import save_csv
from src.drift import detect_feature_drift, summarize_drift
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


def detect_target(df: pd.DataFrame) -> Optional[str]:
    for c in ["default", "target", "y", "label", "class", "status"]:
        if c in df.columns:
            return c
    return None


def preprocess_for_model(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    for id_col in ("Unnamed: 0", "id", "ID", "Id"):
        if id_col in df.columns:
            df = df.drop(columns=[id_col])

    y = df[target]
    # Encode categorical target to numeric
    if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
        y = y.astype('category').cat.codes
    y = y.astype(float)
    X = df.drop(columns=[target])

    # Enhanced feature engineering
    # Basic ratio features for credit datasets
    if 'credit_amount' in X.columns and 'duration' in X.columns:
        X['credit_per_duration'] = X['credit_amount'] / (X['duration'] + 1)
        X['credit_per_duration'] = X['credit_per_duration'].fillna(X['credit_per_duration'].median())

    if 'age' in X.columns and 'credit_amount' in X.columns:
        X['credit_per_age'] = X['credit_amount'] / (X['age'] + 1)
        X['credit_per_age'] = X['credit_per_age'].fillna(X['credit_per_age'].median())

    if 'age' in X.columns and 'duration' in X.columns:
        X['years_to_repay_ratio'] = X['duration'] / (X['age'] + 1)
        X['years_to_repay_ratio'] = X['years_to_repay_ratio'].fillna(X['years_to_repay_ratio'].median())

    # Binning features for better separation
    if 'age' in X.columns:
        X['age_group'] = pd.cut(X['age'], bins=[0, 25, 35, 45, 55, 100], labels=['young', 'young_adult', 'middle', 'senior', 'elder'])
        X['age_group'] = X['age_group'].astype(str)

    if 'duration' in X.columns:
        X['duration_group'] = pd.cut(X['duration'], bins=[0, 12, 24, 36, 60, 1000], labels=['short', 'medium', 'long', 'very_long', 'extreme'])
        X['duration_group'] = X['duration_group'].astype(str)

    if 'credit_amount' in X.columns:
        X['credit_amount_group'] = pd.cut(X['credit_amount'], bins=[0, 1000, 2500, 5000, 10000, 100000], labels=['small', 'medium', 'large', 'very_large', 'huge'])
        X['credit_amount_group'] = X['credit_amount_group'].astype(str)

    # Polynomial features for important numerical columns
    if 'credit_amount' in X.columns:
        X['credit_amount_squared'] = X['credit_amount'] ** 2
    if 'age' in X.columns:
        X['age_squared'] = X['age'] ** 2
    if 'duration' in X.columns:
        X['duration_squared'] = X['duration'] ** 2

    # Fill missing values
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


def tune_and_train_xgb(X_train, y_train, random_state=42):
    """Tune XGBoost hyperparameters with more aggressive search and feature selection."""
    # Split training data for early stopping
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
    )

    # Calculate scale_pos_weight for class imbalance
    pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1]) if len(y_train.unique()) == 2 else 1

    # Base model for initial tuning
    xgb_base = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        random_state=random_state,
        eval_metric='auc',
        early_stopping_rounds=30,
        tree_method='hist'
    )

    # More aggressive hyperparameter grid
    param_grid = {
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.15],
        'n_estimators': [200, 300, 400],
        'min_child_weight': [1, 2, 3],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.2]
    }

    # Randomized search first (faster exploration)
    search = RandomizedSearchCV(
        xgb_base,
        param_distributions=param_grid,
        n_iter=40,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1,
        random_state=random_state,
        verbose=0
    )

    search.fit(X_train_split, y_train_split,
               eval_set=[(X_val, y_val)],
               verbose=False)

    # Get best model and retrain on full training data
    best_params = search.best_params_
    best_model = xgb.XGBClassifier(
        **best_params,
        scale_pos_weight=pos_weight,
        random_state=random_state,
        eval_metric='auc',
        tree_method='hist'
    )
    best_model.fit(X_train, y_train,
                   eval_set=[(X_val, y_val)],
                   verbose=False)

    return best_model


def find_optimal_threshold(y_true, y_prob):
    """Find optimal threshold that maximizes F1-score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_idx = f1_scores.argmax()
    return thresholds[optimal_idx], f1_scores[optimal_idx]


def find_fair_threshold(y_true, y_prob, sensitive, fairness_limit=0.2):
    """
    Find threshold that minimizes bias while keeping accuracy high.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        sensitive: Sensitive attribute values
        fairness_limit: Maximum acceptable demographic parity difference
    
    Returns:
        Optimal threshold that satisfies fairness constraint and maximizes accuracy
    """
    from src.bias_fairness import evaluate_fairness
    
    thresholds = np.linspace(0.1, 0.9, 17)
    
    best_threshold = 0.5
    best_accuracy = 0
    best_bias = 1
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        fairness = evaluate_fairness(y_true, y_pred, sensitive)
        
        bias = abs(fairness.get("demographic_parity_difference", 1))
        
        # Choose threshold that satisfies fairness and maximizes accuracy
        if bias <= fairness_limit and acc > best_accuracy:
            best_accuracy = acc
            best_threshold = t
            best_bias = bias
    
    return best_threshold


def analyze_threshold_tradeoff(y_true, y_prob, sensitive):
    """
    Analyze performance and fairness metrics across multiple thresholds.
    Useful for understanding accuracy-fairness tradeoffs.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        sensitive: Sensitive attribute values
    
    Returns:
        List of dicts with threshold, accuracy, and fairness metrics
    """
    from src.bias_fairness import evaluate_fairness
    
    thresholds = np.linspace(0.1, 0.9, 17)
    results = []
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        fairness = evaluate_fairness(y_true, y_pred, sensitive)
        
        results.append({
            "threshold": float(t),
            "accuracy": float(acc),
            "demographic_parity": float(fairness.get("demographic_parity_difference", 0)),
            "equalized_odds": float(fairness.get("equalized_odds_difference", 0))
        })
    
    return results


def train_models(X_train, y_train, random_state=42, use_smote=True):
    """Train Random Forest and XGBoost models with SMOTE for better imbalance handling."""

    # Apply SMOTE for class imbalance
    if use_smote and len(y_train.unique()) == 2:
        smote = SMOTE(random_state=random_state, k_neighbors=5)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    else:
        X_train_balanced, y_train_balanced = X_train, y_train

    # Random Forest with aggressive tuning and class weights
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1,
        bootstrap=True
    )
    rf_model.fit(X_train_balanced, y_train_balanced)

    # XGBoost with tuning
    xgb_model = tune_and_train_xgb(X_train_balanced, y_train_balanced, random_state)

    return rf_model, xgb_model


def train_and_evaluate_df(df: pd.DataFrame, target: str, sensitive: str, test_size: float = 0.2, seed: int = 42) -> Dict[str, Any]:
    """
    Train and evaluate models on uploaded DataFrame.
    Returns best model and metrics for both models.
    """
    if sensitive not in df.columns:
        raise ValueError(f"Sensitive column '{sensitive}' not found in dataset")
    sensitive_series = df[sensitive]
    
    X, y = preprocess_for_model(df, target)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y if len(y.unique()) > 1 else None)
    
    # Detect feature drift between training and test data
    drift_results = detect_feature_drift(X_train, X_test, p_threshold=0.05)
    drift_summary = summarize_drift(drift_results)

    # Train both models
    rf_model, xgb_model = train_models(X_train, y_train, seed)

    # Create ensemble (voting classifier)
    ensemble = VotingClassifier(
        estimators=[('rf', rf_model), ('xgb', xgb_model)],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)

    models = {'RandomForest': rf_model, 'XGBoost': xgb_model, 'Ensemble': ensemble}
    model_metrics = {}
    
    best_model = None
    best_roc_auc = -1
    
    for name, model in models.items():
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Threshold tuning
        optimal_threshold = 0.5
        tuned_y_pred = y_pred
        if y_prob is not None and len(y_test.unique()) == 2:
            optimal_threshold, max_f1 = find_optimal_threshold(y_test, y_prob)
            tuned_y_pred = (y_prob >= optimal_threshold).astype(int)
        
        # Metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "tuned_accuracy": float(accuracy_score(y_test, tuned_y_pred)),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "tuned_classification_report": classification_report(y_test, tuned_y_pred, output_dict=True)
        }
        if y_prob is not None and len(y_test.unique()) == 2:
            roc_auc = float(roc_auc_score(y_test, y_prob))
            metrics["roc_auc"] = roc_auc
            metrics["optimal_threshold"] = float(optimal_threshold)
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_model = model
        else:
            if best_model is None:
                best_model = model
        
        model_metrics[name] = metrics
    
    # Get predicted probabilities
    y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
    
    # Get sensitive values for test set
    sensitive_test = sensitive_series.loc[X_test.index]
    
    # Find fairness-aware threshold
    fair_threshold = 0.5  # Default value
    if y_prob is not None and len(y_test.unique()) == 2:
        fair_threshold = find_fair_threshold(
            y_test,
            y_prob,
            sensitive_test,
            fairness_limit=0.2
        )
    
    # Final predictions using fairness-aware threshold
    y_pred = (y_prob >= fair_threshold).astype(int) if y_prob is not None else best_model.predict(X_test)
    
    # Fairness evaluation
    from src.bias_fairness import evaluate_fairness
    test_df = X_test.copy()
    test_df['y_true'] = y_test
    test_df['y_pred'] = y_pred
    test_df['sensitive'] = sensitive_test
    fairness = evaluate_fairness(test_df['y_true'], test_df['y_pred'], test_df['sensitive'])
    
    # Threshold tradeoff analysis
    threshold_analysis = []
    if y_prob is not None and len(y_test.unique()) == 2:
        threshold_analysis = analyze_threshold_tradeoff(
            y_test,
            y_prob,
            sensitive_test
        )
    
    result = {
        'best_model': best_model,
        'model_metrics': model_metrics,
        'fairness': fairness,
        'fair_threshold': float(fair_threshold),
        'threshold_analysis': threshold_analysis,
        'drift': drift_results,
        'drift_summary': drift_summary,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    
    return result


def train_and_evaluate(train_path: str, test_path: str, target: str, model_out: str, metrics_out: str, seed: int = 42):
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    if target is None:
        raise ValueError("Target must be specified")

    # Preprocess each separately to avoid index issues
    X_train, y_train = preprocess_for_model(train, target)
    X_test, y_test = preprocess_for_model(test, target)
    
    # Remove any remaining NaN in targets
    train_valid = y_train.notna()
    X_train = X_train[train_valid]
    y_train = y_train[train_valid]
    
    test_valid = y_test.notna()
    X_test = X_test[test_valid]
    y_test = y_test[test_valid]
    
    # Ensure consistent columns between train and test
    common_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    print(f"Train set: {len(X_train)} samples, {len(common_cols)} features")
    print(f"Test set: {len(X_test)} samples")

    # Train both models
    rf_model, xgb_model = train_models(X_train, y_train, seed)

    # Create ensemble (voting classifier)
    ensemble = VotingClassifier(
        estimators=[('rf', rf_model), ('xgb', xgb_model)],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)

    models = {'RandomForest': rf_model, 'XGBoost': xgb_model, 'Ensemble': ensemble}
    model_metrics = {}
    
    best_model = None
    best_roc_auc = -1
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }
        if len(set(y_test.dropna())) == 2:
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
                roc_auc = float(roc_auc_score(y_test, y_prob))
                metrics["roc_auc"] = roc_auc
                
                # Threshold tuning
                optimal_threshold, max_f1 = find_optimal_threshold(y_test, y_prob)
                tuned_y_pred = (y_prob >= optimal_threshold).astype(int)
                metrics["tuned_accuracy"] = float(accuracy_score(y_test, tuned_y_pred))
                metrics["optimal_threshold"] = float(optimal_threshold)
                
                if roc_auc > best_roc_auc:
                    best_roc_auc = roc_auc
                    best_model = model
            except Exception:
                pass
        else:
            if best_model is None:
                best_model = model
        
        model_metrics[name] = metrics

    # Save best model
    model_path = Path(model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, str(model_path.with_suffix('.pkl')))

    # Save metrics (include both models' metrics)
    best_model_name = 'Ensemble' if isinstance(best_model, VotingClassifier) else \
                      'XGBoost' if isinstance(best_model, xgb.XGBClassifier) else 'RandomForest'
    metrics_dict = {
        'best_model': best_model_name,
        'best_roc_auc': best_roc_auc,
        'model_metrics': model_metrics
    }
    save_json(metrics_dict, Path(metrics_out))

    preds_path = Path('reports/model_predictions.csv')
    try:
        y_pred = best_model.predict(X_test)
        preds_df = X_test.copy()
        preds_df['y_true'] = list(y_test)
        preds_df['y_pred'] = list(y_pred)
        preds_df = preds_df.reset_index(drop=True)
        save_csv(preds_df, preds_path)
        print(f"Saved predictions to {preds_path}")
    except Exception:
        pass

    print(f"Best model ({best_model_name}) saved to {model_path}")
    print(f"Metrics saved to {metrics_out}")


def main():
    import argparse

    # Add project root to Python path
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.append(str(ROOT))

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
