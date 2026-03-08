"""Streamlit dashboard for ML Governance Toolkit."""
from __future__ import annotations
from pathlib import Path
import json
from datetime import datetime
from typing import Any, Dict, List
import sys

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.model_training import train_and_evaluate_df, detect_target
from src.explainability_layer import explain_model
from src.utils.evidence_pack import build_audit_bundle, save_audit_json, save_audit_pdf
from src.bias_fairness import evaluate_fairness

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
HISTORY_FILE = REPORTS / "model_history.json"


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------

def load_history() -> List[Dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_history(history: List[Dict[str, Any]]):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


# ---------------------------------------------------------------------------
# Bias mitigation simulation
# ---------------------------------------------------------------------------

def simulate_bias_mitigation(df: pd.DataFrame, target: str, sensitive: str) -> Dict[str, Any]:
    """Simulate reweighting to mitigate bias (uses feature-selected pipeline for fair comparison)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score
    from src.model_training import preprocess_for_model
    from sklearn.feature_selection import SelectFromModel

    X, y = preprocess_for_model(df, target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature selection consistent with main pipeline
    sel_rf = RandomForestClassifier(
        n_estimators=100, max_depth=8,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    sel_rf.fit(X_train, y_train)
    from sklearn.feature_selection import SelectFromModel
    selector = SelectFromModel(sel_rf, threshold="median", prefit=True)
    selected_cols = X_train.columns[selector.get_support()].tolist()
    X_train_sel = X_train[selected_cols]
    X_test_sel  = X_test[selected_cols]

    # Reweighting by inverse sensitive-group frequency
    sensitive_train = df.loc[X_train.index, sensitive]
    group_counts = sensitive_train.value_counts()
    total = len(sensitive_train)
    sample_weight = sensitive_train.map(lambda x: total / group_counts[x])

    model = RandomForestClassifier(
        n_estimators=200, class_weight="balanced_subsample", random_state=42
    )
    model.fit(X_train_sel, y_train, sample_weight=sample_weight)

    y_pred = model.predict(X_test_sel)
    y_prob = model.predict_proba(X_test_sel)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc":  float(roc_auc_score(y_test, y_prob)),
    }

    sensitive_test = df.loc[X_test.index, sensitive]
    fairness = evaluate_fairness(y_test, y_pred, sensitive_test)
    metrics.update(fairness)
    return metrics


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

st.title("ML Governance Toolkit Dashboard")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")

else:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    columns = list(df.columns)
    detected_target = detect_target(df)

    target_col = st.selectbox(
        "Select Target Column",
        columns,
        index=columns.index(detected_target) if detected_target and detected_target in columns else 0,
    )
    sensitive_col = st.selectbox("Select Sensitive Attribute (for Fairness)", columns)

    if st.button("Run Analysis"):
        with st.spinner("Running analysis... this may take a minute."):

            # ----------------------------------------------------------------
            # Train & evaluate
            # ----------------------------------------------------------------
            result = train_and_evaluate_df(df, target_col, sensitive_col)

            # Unpack — use new keys from updated model_training.py
            model           = result["best_model"]
            best_name       = result["best_model_name"]   # ✅ Fixed: use returned name
            model_metrics   = result["model_metrics"]
            fairness        = result["fairness"]
            fair_threshold  = result.get("fair_threshold", 0.5)
            threshold_analysis = result.get("threshold_analysis", [])
            drift_results   = result.get("drift", {})
            drift_summary   = result.get("drift_summary", {})
            cv_auc_mean     = result.get("cv_auc_mean", None)   # ✅ New
            cv_auc_std      = result.get("cv_auc_std",  None)   # ✅ New
            X_train         = result["X_train"]   # feature-selected
            X_test          = result["X_test"]    # feature-selected ✅
            y_train         = result["y_train"]
            y_test          = result["y_test"]
            y_pred          = result["y_pred"]
            y_prob          = result["y_prob"]

            # Safe best_metrics lookup with fallback
            best_metrics = model_metrics.get(
                best_name,
                model_metrics.get("Ensemble", list(model_metrics.values())[0])
            )

            # ----------------------------------------------------------------
            # Explainability — X_test is already feature-selected
            # ----------------------------------------------------------------
            shap_features = explain_model(model, X_test)

            # ----------------------------------------------------------------
            # Model Performance
            # ----------------------------------------------------------------
            st.header("Model Performance")

            # Cross-validation AUC (honest estimate) ✅ New section
            if cv_auc_mean is not None:
                st.subheader("Cross-Validation AUC (5-fold, pre-training estimate)")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean CV AUC", f"{cv_auc_mean:.3f}")
                with col2:
                    st.metric("CV AUC Std Dev", f"± {cv_auc_std:.3f}")
                st.caption(
                    "This is a more reliable performance estimate than a single train/test split, "
                    "especially on small datasets like German Credit."
                )

            # Model comparison table
            st.subheader("Model Comparison")
            comparison_rows = []
            for name, m in model_metrics.items():
                comparison_rows.append({
                    "Model":        name,
                    "Accuracy":     round(m.get("accuracy", 0), 4),
                    "Tuned Acc":    round(m.get("tuned_accuracy", m.get("accuracy", 0)), 4),
                    "ROC AUC":      round(m.get("roc_auc", 0), 4),
                    "Best ✓":       "✓" if name == best_name else "",
                })
            st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Best Model", best_name)
                st.metric("Accuracy", f"{best_metrics.get('accuracy', 0):.3f}")
            with col2:
                if "roc_auc" in best_metrics:
                    st.metric("ROC AUC", f"{best_metrics['roc_auc']:.3f}")
                if "tuned_accuracy" in best_metrics:
                    st.metric("Tuned Accuracy", f"{best_metrics['tuned_accuracy']:.3f}")

            # Threshold tuning
            if "optimal_threshold" in best_metrics:
                st.subheader("Threshold Tuning")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Optimal Threshold (F1)", f"{best_metrics['optimal_threshold']:.3f}")
                with col2:
                    st.metric("Fairness-Aware Threshold", f"{fair_threshold:.3f}")
                with col3:
                    st.metric("Tuned Accuracy", f"{best_metrics.get('tuned_accuracy', 0):.3f}")

            # Threshold tradeoff charts
            if threshold_analysis:
                st.subheader("Threshold Tradeoff Analysis")
                ta_df = pd.DataFrame(threshold_analysis)

                col1, col2, col3 = st.columns(3)

                with col1:
                    fig_acc = go.Figure()
                    fig_acc.add_trace(go.Scatter(
                        x=ta_df["threshold"], y=ta_df["accuracy"],
                        mode="lines+markers", name="Accuracy",
                        line=dict(color="blue")
                    ))
                    fig_acc.add_vline(
                        x=fair_threshold, line_dash="dash",
                        line_color="red", annotation_text="Fair Threshold"
                    )
                    fig_acc.update_layout(
                        title="Accuracy vs Threshold",
                        xaxis_title="Threshold", yaxis_title="Accuracy", height=400
                    )
                    st.plotly_chart(fig_acc, use_container_width=True)

                with col2:
                    fig_dp = go.Figure()
                    fig_dp.add_trace(go.Scatter(
                        x=ta_df["threshold"], y=ta_df["demographic_parity"],
                        mode="lines+markers", name="Demographic Parity",
                        line=dict(color="orange")
                    ))
                    fig_dp.add_hline(y=0.2, line_dash="dash", line_color="red",  annotation_text="Fairness Limit")
                    fig_dp.add_vline(x=fair_threshold, line_dash="dash", line_color="green", annotation_text="Fair Threshold")
                    fig_dp.update_layout(
                        title="Demographic Parity vs Threshold",
                        xaxis_title="Threshold", yaxis_title="DP Difference", height=400
                    )
                    st.plotly_chart(fig_dp, use_container_width=True)

                with col3:
                    fig_eo = go.Figure()
                    fig_eo.add_trace(go.Scatter(
                        x=ta_df["threshold"], y=ta_df["equalized_odds"],
                        mode="lines+markers", name="Equalized Odds",
                        line=dict(color="purple")
                    ))
                    fig_eo.add_hline(y=0.2, line_dash="dash", line_color="red",  annotation_text="Fairness Limit")
                    fig_eo.add_vline(x=fair_threshold, line_dash="dash", line_color="green", annotation_text="Fair Threshold")
                    fig_eo.update_layout(
                        title="Equalized Odds vs Threshold",
                        xaxis_title="Threshold", yaxis_title="EO Difference", height=400
                    )
                    st.plotly_chart(fig_eo, use_container_width=True)

                st.subheader("Threshold Analysis Details")
                st.dataframe(ta_df.round(4), use_container_width=True)

            # ----------------------------------------------------------------
            # Data Drift
            # ----------------------------------------------------------------
            st.header("Data Drift Detection")
            if drift_summary:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Features Monitored",  drift_summary.get("total_features_monitored", 0))
                with col2:
                    st.metric("Features with Drift", drift_summary.get("features_with_drift", 0))
                with col3:
                    st.metric("Drift %", f"{drift_summary.get('drift_percentage', 0):.1f}%")
                with col4:
                    has_drift = drift_summary.get("features_with_drift", 0) > 0
                    st.metric("Status", "⚠️ Drift Detected" if has_drift else "✅ No Drift")

                if drift_summary.get("drifted_features"):
                    st.subheader("Drifted Features Details")
                    drift_rows = []
                    for feature in drift_summary.get("drifted_features", []):
                        if feature in drift_results:
                            drift_rows.append({
                                "Feature":      feature,
                                "P-Value":      f"{drift_results[feature]['p_value']:.6f}",
                                "KS Statistic": f"{drift_results[feature].get('ks_statistic', 0):.4f}",
                                "Drift":        "✓",
                            })
                    if drift_rows:
                        st.dataframe(pd.DataFrame(drift_rows), use_container_width=True)

            # ----------------------------------------------------------------
            # Fairness Metrics
            # ----------------------------------------------------------------
            st.header("Fairness Metrics")
            col1, col2 = st.columns(2)
            with col1:
                dp = fairness["demographic_parity_difference"]
                st.metric(
                    "Demographic Parity Diff", f"{dp:.3f}",
                    delta="⚠️ High" if abs(dp) > 0.2 else "✅ OK",
                    delta_color="inverse"
                )
            with col2:
                eo = fairness["equalized_odds_difference"]
                st.metric(
                    "Equalized Odds Diff", f"{eo:.3f}",
                    delta="⚠️ High" if abs(eo) > 0.2 else "✅ OK",
                    delta_color="inverse"
                )

            # ----------------------------------------------------------------
            # SHAP Explainability
            # ----------------------------------------------------------------
            st.header("Top SHAP Features")
            shap_df = pd.DataFrame(
                list(shap_features.items()), columns=["Feature", "Mean |SHAP|"]
            ).sort_values("Mean |SHAP|", ascending=False)
            st.bar_chart(shap_df.set_index("Feature"))

            # ----------------------------------------------------------------
            # Save reports
            # ----------------------------------------------------------------
            REPORTS.mkdir(parents=True, exist_ok=True)

            # Model metrics
            save_metrics = {
                k: v for k, v in best_metrics.items()
                if k in ["accuracy", "tuned_accuracy", "classification_report",
                         "roc_auc", "optimal_threshold"]
            }
            with open(REPORTS / "model_metrics.json", "w") as f:
                json.dump(save_metrics, f, indent=2)

            # Explainability
            with open(REPORTS / "explainability_summary.json", "w") as f:
                json.dump({"mean_abs_shap": shap_features}, f, indent=2)

            # Fairness
            with open(REPORTS / "fairness_summary.json", "w") as f:
                json.dump(fairness, f, indent=2)

            # Dataset profile
            from src.utils.profile import profile as profile_fn
            profile_data = profile_fn(df)
            with open(REPORTS / "dataset_profile.json", "w") as f:
                json.dump(profile_data, f, indent=2)

            # Predictions — bin continuous sensitive attrs before saving
            preds_df = X_test.copy()
            preds_df["y_true"] = y_test.values
            preds_df["y_pred"] = y_pred
            raw_sensitive = pd.Series(df.loc[X_test.index, sensitive_col].values)

            # ✅ Bin numeric sensitive columns so trajectory shows meaningful groups
            if pd.api.types.is_numeric_dtype(raw_sensitive) and raw_sensitive.nunique() > 10:
                preds_df["sensitive"] = pd.cut(
                    raw_sensitive,
                    bins=[0, 25, 35, 50, 100],
                    labels=["young", "young_adult", "middle_aged", "senior"]
                ).astype(str)
            else:
                preds_df["sensitive"] = raw_sensitive.values
            preds_df.to_csv(REPORTS / "model_predictions.csv", index=False)

            # Trajectory summary
            from src.trajectory.analysis import performance_by_bucket
            trajectory_data = {"psi": {}, "bucket_performance": [], "cohort_over_time": []}
            try:
                perf_bucket = performance_by_bucket(preds_df, "sensitive")
                trajectory_data["bucket_performance"] = perf_bucket.to_dict(orient="records")
            except Exception:
                pass
            with open(REPORTS / "trajectory_summary.json", "w") as f:
                json.dump(trajectory_data, f, indent=2)

            # History entry — ✅ Uses best_name directly
            history = load_history()
            history.append({
                "timestamp":       datetime.utcnow().isoformat() + "Z",
                "best_model":      best_name,
                "roc_auc":         best_metrics.get("roc_auc"),
                "accuracy":        best_metrics.get("accuracy"),
                "tuned_accuracy":  best_metrics.get("tuned_accuracy"),
                "optimal_threshold": best_metrics.get("optimal_threshold"),
                "fair_threshold":  fair_threshold,
                "cv_auc_mean":     cv_auc_mean,   # ✅ Now logged
                "cv_auc_std":      cv_auc_std,
                "fairness_dp":     fairness["demographic_parity_difference"],
                "fairness_eo":     fairness["equalized_odds_difference"],
                "drift_features":  drift_summary.get("features_with_drift", 0),
                "drift_percentage": drift_summary.get("drift_percentage", 0.0),
                "top_shap":        list(shap_features.keys())[:5],
            })
            save_history(history)

            # Audit bundle
            missing = [
                f for f in [
                    "model_metrics.json", "fairness_summary.json",
                    "explainability_summary.json", "trajectory_summary.json",
                    "model_history.json",
                ]
                if not (REPORTS / f).exists()
            ]
            if missing:
                st.warning(f"Audit may be incomplete. Missing: {missing}")
            else:
                st.info("All audit artifacts present. Generating audit report...")

            audit_bundle = build_audit_bundle(REPORTS)
            save_audit_json(audit_bundle, REPORTS / "audit_report.json")
            save_audit_pdf(audit_bundle,  REPORTS / "audit_report.pdf")
            st.success(f"Audit reports saved to {REPORTS}")

            # ----------------------------------------------------------------
            # Bias mitigation feedback loop
            # ----------------------------------------------------------------
            if abs(fairness["demographic_parity_difference"]) > 0.2:
                st.warning("High bias detected! Simulating bias mitigation via reweighting...")
                mitigated = simulate_bias_mitigation(df, target_col, sensitive_col)
                st.subheader("After Mitigation (Reweighting)")
                col1, col2, col3 = st.columns(3)
                orig_auc = best_metrics.get("roc_auc", 0)
                orig_dp  = fairness["demographic_parity_difference"]
                orig_eo  = fairness["equalized_odds_difference"]
                with col1:
                    st.metric("ROC AUC", f"{mitigated['roc_auc']:.3f}",
                              delta=f"{mitigated['roc_auc'] - orig_auc:+.3f}")
                with col2:
                    st.metric("DP Diff",  f"{mitigated['demographic_parity_difference']:.3f}",
                              delta=f"{mitigated['demographic_parity_difference'] - orig_dp:+.3f}",
                              delta_color="inverse")
                with col3:
                    st.metric("EO Diff",  f"{mitigated['equalized_odds_difference']:.3f}",
                              delta=f"{mitigated['equalized_odds_difference'] - orig_eo:+.3f}",
                              delta_color="inverse")


# ---------------------------------------------------------------------------
# Trajectory section (always visible, even before a run)
# ---------------------------------------------------------------------------
st.header("Trajectory Analysis")
history = load_history()

if not history:
    st.info("No history available yet. Run analysis to start tracking.")
else:
    hist_df = pd.DataFrame(history)
    hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])

    # ROC AUC over time
    if "roc_auc" in hist_df.columns and hist_df["roc_auc"].notna().any():
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_df["timestamp"], y=hist_df["roc_auc"],
            mode="lines+markers", name="ROC AUC", line=dict(color="steelblue")
        ))
        # ✅ Show CV AUC band if available
        if "cv_auc_mean" in hist_df.columns and hist_df["cv_auc_mean"].notna().any():
            fig.add_trace(go.Scatter(
                x=hist_df["timestamp"], y=hist_df["cv_auc_mean"],
                mode="lines+markers", name="CV AUC (mean)",
                line=dict(color="gray", dash="dot")
            ))
        fig.update_layout(
            title="ROC AUC Over Time",
            xaxis_title="Time", yaxis_title="AUC"
        )
        st.plotly_chart(fig)

    # Fairness over time
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=hist_df["timestamp"], y=hist_df["fairness_dp"],
        mode="lines+markers", name="Demographic Parity Diff", line=dict(color="orange")
    ))
    fig2.add_trace(go.Scatter(
        x=hist_df["timestamp"], y=hist_df["fairness_eo"],
        mode="lines+markers", name="Equalized Odds Diff", line=dict(color="purple")
    ))
    fig2.add_hline(y=0.2,  line_dash="dash", line_color="red", annotation_text="Fairness Limit")
    fig2.add_hline(y=-0.2, line_dash="dash", line_color="red")
    fig2.update_layout(
        title="Fairness Metrics Over Time",
        xaxis_title="Time", yaxis_title="Difference"
    )
    st.plotly_chart(fig2)

    st.subheader("Recent Runs")
    display_cols = [
        c for c in [
            "timestamp", "best_model", "roc_auc", "cv_auc_mean",
            "accuracy", "fair_threshold", "fairness_dp", "fairness_eo",
            "drift_features"
        ]
        if c in hist_df.columns
    ]
    st.dataframe(hist_df[display_cols].tail(10), use_container_width=True)