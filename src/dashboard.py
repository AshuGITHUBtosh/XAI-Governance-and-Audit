"""Streamlit dashboard for ML Governance Toolkit."""
from __future__ import annotations
from pathlib import Path
import json
from datetime import datetime
from typing import Any, Dict, List
import os
import sys
import sys
from pathlib import Path

# Add project root (explainable_ai) to Python path
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
REPORTS = ROOT / 'reports'
HISTORY_FILE = REPORTS / 'model_history.json'


def load_history() -> List[Dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return []
    with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_history(history: List[Dict[str, Any]]):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)


def simulate_bias_mitigation(df: pd.DataFrame, target: str, sensitive: str) -> Dict[str, Any]:
    """
    Simulate reweighting to mitigate bias.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score
    from src.model_training import preprocess_for_model
    from src.bias_fairness import evaluate_fairness
    
    X, y = preprocess_for_model(df, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Simple reweighting: weight by inverse frequency of sensitive group
    sensitive_train = df.loc[X_train.index, sensitive]
    group_counts = sensitive_train.value_counts()
    total = len(sensitive_train)
    sample_weight = sensitive_train.map(lambda x: total / group_counts[x])
    
    # Train with weights
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob))
    }
    
    test_df = X_test.copy()
    test_df['y_true'] = y_test
    test_df['y_pred'] = y_pred
    test_df['sensitive'] = df.loc[X_test.index, sensitive]
    fairness = evaluate_fairness(test_df['y_true'], test_df['y_pred'], test_df['sensitive'])
    metrics.update(fairness)
    
    return metrics


# Main dashboard code
st.title('ML Governance Toolkit Dashboard')

# File upload
uploaded_file = st.file_uploader("Upload CSV Dataset", type=['csv'])
if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
else:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df.head())
    
    # Column selection
    columns = list(df.columns)
    target_col = st.selectbox("Select Target Column", columns, index=columns.index(detect_target(df)) if detect_target(df) else 0)
    sensitive_col = st.selectbox("Select Sensitive Attribute (for Fairness)", columns)
    
    if st.button("Run Analysis"):
        with st.spinner("Running analysis..."):
            # Train and evaluate
            result = train_and_evaluate_df(df, target_col, sensitive_col)
            model = result['best_model']
            model_metrics = result['model_metrics']
            fairness = result['fairness']
            fair_threshold = result.get('fair_threshold', 0.5)
            threshold_analysis = result.get('threshold_analysis', [])
            drift_results = result.get('drift', {})
            drift_summary = result.get('drift_summary', {})
            X_test = result['X_test']
            y_test = result['y_test']
            y_pred = result['y_pred']
            y_prob = result['y_prob']
            
            # Explainability
            shap_features = explain_model(model, X_test)
            
            # Display results
            st.header("Model Performance")
            
            # Show metrics for both models
            st.subheader("Model Comparison")
            comparison_df = pd.DataFrame({
                'Model': list(model_metrics.keys()),
                'Accuracy': [m['accuracy'] for m in model_metrics.values()],
                'ROC AUC': [m.get('roc_auc', 'N/A') for m in model_metrics.values()]
            })
            st.dataframe(comparison_df)
            
            # Best model metrics
            best_name = 'XGBoost' if hasattr(model, 'get_booster') else 'RandomForest'
            best_metrics = model_metrics[best_name]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Best Model Accuracy", f"{best_metrics['accuracy']:.3f}")
            with col2:
                if 'roc_auc' in best_metrics:
                    st.metric("Best Model ROC AUC", f"{best_metrics['roc_auc']:.3f}")
            
            # Show threshold tuning info
            if 'optimal_threshold' in best_metrics:
                st.subheader("Threshold Tuning")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Optimal Threshold (F1)", f"{best_metrics['optimal_threshold']:.3f}")
                with col2:
                    st.metric("Fairness-Aware Threshold", f"{fair_threshold:.3f}")
                with col3:
                    st.metric("Tuned Accuracy", f"{best_metrics['tuned_accuracy']:.3f}")
            
            # Threshold Tradeoff Analysis
            if threshold_analysis:
                st.subheader("Threshold Tradeoff Analysis")
                
                # Convert to DataFrame for easier plotting
                ta_df = pd.DataFrame(threshold_analysis)
                
                # Create three columns for the charts
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Accuracy vs Threshold
                    fig_acc = go.Figure()
                    fig_acc.add_trace(go.Scatter(
                        x=ta_df['threshold'],
                        y=ta_df['accuracy'],
                        mode='lines+markers',
                        name='Accuracy',
                        line=dict(color='blue')
                    ))
                    fig_acc.add_vline(x=fair_threshold, line_dash="dash", line_color="red", annotation_text="Fair Threshold")
                    fig_acc.update_layout(
                        title='Accuracy vs Threshold',
                        xaxis_title='Threshold',
                        yaxis_title='Accuracy',
                        height=400
                    )
                    st.plotly_chart(fig_acc, use_container_width=True)
                
                with col2:
                    # Demographic Parity vs Threshold
                    fig_dp = go.Figure()
                    fig_dp.add_trace(go.Scatter(
                        x=ta_df['threshold'],
                        y=ta_df['demographic_parity'],
                        mode='lines+markers',
                        name='Demographic Parity',
                        line=dict(color='orange')
                    ))
                    fig_dp.add_hline(y=0.2, line_dash="dash", line_color="red", annotation_text="Fairness Limit")
                    fig_dp.add_vline(x=fair_threshold, line_dash="dash", line_color="green", annotation_text="Fair Threshold")
                    fig_dp.update_layout(
                        title='Demographic Parity Bias vs Threshold',
                        xaxis_title='Threshold',
                        yaxis_title='Demographic Parity Difference',
                        height=400
                    )
                    st.plotly_chart(fig_dp, use_container_width=True)
                
                with col3:
                    # Equalized Odds vs Threshold
                    fig_eo = go.Figure()
                    fig_eo.add_trace(go.Scatter(
                        x=ta_df['threshold'],
                        y=ta_df['equalized_odds'],
                        mode='lines+markers',
                        name='Equalized Odds',
                        line=dict(color='purple')
                    ))
                    fig_eo.add_hline(y=0.2, line_dash="dash", line_color="red", annotation_text="Fairness Limit")
                    fig_eo.add_vline(x=fair_threshold, line_dash="dash", line_color="green", annotation_text="Fair Threshold")
                    fig_eo.update_layout(
                        title='Equalized Odds Bias vs Threshold',
                        xaxis_title='Threshold',
                        yaxis_title='Equalized Odds Difference',
                        height=400
                    )
                    st.plotly_chart(fig_eo, use_container_width=True)
                
                # Show threshold analysis table
                st.subheader("Threshold Analysis Details")
                st.dataframe(ta_df.round(4))
            
            # Data Drift Detection
            st.header("Data Drift Detection")
            if drift_summary:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Features Monitored", drift_summary.get('total_features_monitored', 0))
                with col2:
                    st.metric("Features with Drift", drift_summary.get('features_with_drift', 0))
                with col3:
                    st.metric("Drift Percentage", f"{drift_summary.get('drift_percentage', 0):.1f}%")
                with col4:
                    st.metric("Status", "⚠️ Drift Detected" if drift_summary.get('features_with_drift', 0) > 0 else "✅ No Drift")
                
                # Show drifted features
                if drift_summary.get('drifted_features'):
                    st.subheader("Drifted Features Details")
                    drift_detail_data = []
                    for feature in drift_summary.get('drifted_features', []):
                        if feature in drift_results:
                            drift_detail_data.append({
                                'Feature': feature,
                                'P-Value': f"{drift_results[feature]['p_value']:.6f}",
                                'KS Statistic': f"{drift_results[feature].get('ks_statistic', 0):.4f}",
                                'Drift Detected': '✓'
                            })
                    if drift_detail_data:
                        drift_df = pd.DataFrame(drift_detail_data)
                        st.dataframe(drift_df, use_container_width=True)
            
            st.header("Fairness Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Demographic Parity Diff", f"{fairness['demographic_parity_difference']:.3f}")
            with col2:
                st.metric("Equalized Odds Diff", f"{fairness['equalized_odds_difference']:.3f}")
            
            st.header("Top SHAP Features")
            shap_df = pd.DataFrame(list(shap_features.items()), columns=['Feature', 'Importance'])
            st.bar_chart(shap_df.set_index('Feature'))
            
            # Save individual reports
            # Model metrics
            model_metrics_to_save = {k: v for k, v in best_metrics.items() if k in ['accuracy', 'tuned_accuracy', 'classification_report', 'roc_auc', 'optimal_threshold']}
            with open(REPORTS / 'model_metrics.json', 'w') as f:
                json.dump(model_metrics_to_save, f, indent=2)
            
            # Explainability
            explainability_data = {'mean_abs_shap': shap_features}
            with open(REPORTS / 'explainability_summary.json', 'w') as f:
                json.dump(explainability_data, f, indent=2)
            
            # Fairness
            with open(REPORTS / 'fairness_summary.json', 'w') as f:
                json.dump(fairness, f, indent=2)
            
            # Dataset profile
            from src.utils.profile import profile
            profile_data = profile(df)
            with open(REPORTS / 'dataset_profile.json', 'w') as f:
                json.dump(profile_data, f, indent=2)
            
            # Model predictions
            preds_df = X_test.copy()
            preds_df['y_true'] = y_test
            preds_df['y_pred'] = y_pred
            preds_df['sensitive'] = df.loc[X_test.index, sensitive_col]
            preds_df.to_csv(REPORTS / 'model_predictions.csv', index=False)
            
            # Trajectory summary
            from src.trajectory.analysis import performance_by_bucket
            trajectory_data = {'psi': {}, 'bucket_performance': [], 'cohort_over_time': []}
            if sensitive_col in preds_df.columns:
                try:
                    perf_bucket = performance_by_bucket(preds_df, sensitive_col)
                    trajectory_data['bucket_performance'] = perf_bucket.to_dict(orient='records')
                except Exception:
                    pass  # If fails, keep empty
            with open(REPORTS / 'trajectory_summary.json', 'w') as f:
                json.dump(trajectory_data, f, indent=2)

            # Log to history
            history = load_history()
            run_entry = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'roc_auc': best_metrics.get('roc_auc', None),
                'accuracy': best_metrics.get('accuracy', None),
                'tuned_accuracy': best_metrics.get('tuned_accuracy', None),
                'optimal_threshold': best_metrics.get('optimal_threshold', None),
                'fair_threshold': fair_threshold,
                'fairness_dp': fairness['demographic_parity_difference'],
                'fairness_eo': fairness['equalized_odds_difference'],
                'drift_features': drift_summary.get('features_with_drift', 0),
                'drift_percentage': drift_summary.get('drift_percentage', 0.0),
                'top_shap': list(shap_features.keys())[:5],
                'best_model': best_name
            }
            history.append(run_entry)

            # 1. Save history FIRST
            save_history(history)

            # 2. Ensure all required artifacts exist BEFORE audit
            required_files = [
                'model_metrics.json',
                'fairness_summary.json',
                'explainability_summary.json',
                'trajectory_summary.json',
                'model_history.json'
            ]

            missing = [f for f in required_files if not (REPORTS / f).exists()]

            if missing:
                st.warning(f"Audit may be incomplete. Missing files: {missing}")
            else:
                st.info("All audit artifacts present. Generating audit report...")

            # 3. Generate audit AFTER validation
            audit_bundle = build_audit_bundle(REPORTS)
            audit_json = REPORTS / 'audit_report.json'
            audit_pdf = REPORTS / 'audit_report.pdf'

            save_audit_json(audit_bundle, audit_json)
            save_audit_pdf(audit_bundle, audit_pdf)

            st.success(f"Audit reports saved to {audit_json} and {audit_pdf}")
            
            # Feedback loop
            if fairness['demographic_parity_difference'] > 0.2:
                st.warning("High bias detected! Simulating bias mitigation...")
                mitigated_metrics = simulate_bias_mitigation(df, target_col, sensitive_col)
                st.subheader("After Mitigation (Reweighting)")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("ROC AUC", f"{mitigated_metrics['roc_auc']:.3f}", delta=f"{mitigated_metrics['roc_auc'] - best_metrics.get('roc_auc', 0):.3f}")
                with col2:
                    st.metric("DP Diff", f"{mitigated_metrics['demographic_parity_difference']:.3f}", delta=f"{mitigated_metrics['demographic_parity_difference'] - fairness['demographic_parity_difference']:.3f}")
                with col3:
                    st.metric("EO Diff", f"{mitigated_metrics['equalized_odds_difference']:.3f}", delta=f"{mitigated_metrics['equalized_odds_difference'] - fairness['equalized_odds_difference']:.3f}")

# Trajectory section
st.header("Trajectory Analysis")
history = load_history()
if history:
    hist_df = pd.DataFrame(history)
    hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'])
    
    # ROC AUC over time
    if 'roc_auc' in hist_df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_df['timestamp'], y=hist_df['roc_auc'], mode='lines+markers', name='ROC AUC'))
        fig.update_layout(title='ROC AUC Over Time', xaxis_title='Time', yaxis_title='ROC AUC')
        st.plotly_chart(fig)
    
    # Fairness over time
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=hist_df['timestamp'], y=hist_df['fairness_dp'], mode='lines+markers', name='Demographic Parity Diff'))
    fig2.add_trace(go.Scatter(x=hist_df['timestamp'], y=hist_df['fairness_eo'], mode='lines+markers', name='Equalized Odds Diff'))
    fig2.update_layout(title='Fairness Metrics Over Time', xaxis_title='Time', yaxis_title='Difference')
    st.plotly_chart(fig2)
    
    st.subheader("Recent Runs")
    st.dataframe(hist_df.tail(5))
else:
    st.info("No history available. Run analysis to start tracking.")

