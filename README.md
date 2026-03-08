# Explainable AI Governance & Model Risk Management System

A comprehensive framework for transparent, accountable, and continuously monitored machine learning models in financial risk management. Built with SHAP explainability, fairness assessment, drift detection, and automated audit reporting.

---

## Project Overview

This system evaluates and monitors ML models used in financial risk prediction. It integrates:

- **Model Training** — Random Forest, XGBoost, and Ensemble (Voting Classifier) with SMOTE balancing and feature selection
- **Explainability** — SHAP (SHapley Additive Explanations) for feature importance
- **Fairness Assessment** — Demographic Parity Difference and Equalized Odds Difference across sensitive attributes
- **Threshold Optimization** — Fairness-aware threshold tuning balancing accuracy and bias
- **Drift Detection** — Kolmogorov-Smirnov test for feature distribution shift
- **Trajectory Monitoring** — Tracks model performance and fairness metrics across runs over time
- **Audit Reporting** — Automated PDF and JSON audit reports for regulatory review
- **Interactive Dashboard** — Streamlit-based UI for end-to-end pipeline execution

---

## Project Structure

```
explainable_ai/
├── data/
│   ├── raw/                    # Raw CSV datasets
│   └── processed/              # Train/test parquet files
├── src/
│   ├── utils/
│   │   ├── data.py             # Data ingestion and preprocessing
│   │   ├── evidence_pack.py    # Audit report generation (PDF + JSON)
│   │   ├── io.py               # File I/O utilities
│   │   └── profile.py          # Dataset profiling
│   ├── trajectory/
│   │   ├── analysis.py         # PSI, cohort, bucket performance analysis
│   │   └── run_analysis.py     # Trajectory summary runner
│   ├── bias_fairness.py        # Fairness metrics (DP, EO)
│   ├── dashboard.py            # Streamlit dashboard
│   ├── drift.py                # Feature drift detection
│   ├── explainability_layer.py # SHAP explainability
│   └── model_training.py       # Model training and evaluation pipeline
├── reports/                    # Generated reports and summaries
├── artifacts/                  # Saved model files
├── notebooks/                  # Jupyter exploration notebooks
├── requirements.txt
└── README.md
```

---

## Setup

```powershell
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Running the Pipeline

### Option A — Dashboard (Recommended)

The dashboard runs the full pipeline automatically in one click.

```powershell
# 1. Ingest and preprocess raw data
python -m src.utils.data --raw data/raw/credit.csv --out data/processed

# 2. Launch dashboard
streamlit run src/dashboard.py
```

Then in the dashboard:
1. Upload `data/raw/credit.csv`
2. Select **Target Column:** `default`
3. Select **Sensitive Attribute:** `personal_status` (or `age`, `foreign_worker`, `job`)
4. Click **Run Analysis**

---

### Option B — Manual Step-by-Step (CLI)

Run each module individually for debugging or custom workflows.

```powershell
# Step 1 — Ingest and preprocess raw CSV (handles good/bad encoding automatically)
python -m src.utils.data --raw data/raw/credit.csv --out data/processed

# Step 2 — Verify target encoding is correct
python -c "import pandas as pd; print(pd.read_parquet('data/processed/train.parquet')['default'].value_counts())"
# Expected: 0.0 (good): 560, 1.0 (bad): 240

# Step 3 — Train models (RF + XGBoost + Ensemble)
python src/model_training.py \
    --train data/processed/train.parquet \
    --test data/processed/test.parquet \
    --target default \
    --model artifacts/model_xgb.json \
    --metrics reports/model_metrics.json

# Step 4 — Generate dataset profile
python src/utils/profile.py \
    --train data/processed/train.parquet \
    --out reports/dataset_profile.json

# Step 5 — Generate SHAP explainability summary
python src/explainability_layer.py \
    --model artifacts/model_xgb.pkl \
    --data data/processed/test.parquet \
    --out reports/explainability_summary.json

# Step 6 — Compute fairness metrics
python src/bias_fairness.py \
    --preds reports/model_predictions.csv \
    --sensitive personal_status \
    --y_true y_true \
    --y_pred y_pred \
    --out reports/fairness_summary.json

# Step 7 — Run trajectory analysis
python -m src.trajectory.run_analysis --sensitive personal_status

# Step 8 — Generate audit report (PDF + JSON)
python src/utils/evidence_pack.py
```

---

## Dataset

**German Credit Dataset** (UCI Machine Learning Repository)

- 1,000 loan applicants, 20 features, binary target (`good`/`bad` credit)
- 70/30 class imbalance (700 good, 300 bad)
- Stratified 80/20 train/test split

### Recommended Sensitive Attributes

Run the pipeline with different sensitive attributes to demonstrate governance across multiple protected characteristics:

| Sensitive Attribute | Groups | Regulatory Relevance |
|---|---|---|
| `personal_status` | male single, female div/dep/mar, male div/sep, male mar/wid | Gender + marital status discrimination |
| `age` | young, young_adult, middle_aged, senior | Age discrimination in credit |
| `foreign_worker` | yes, no | Nationality-based discrimination |
| `job` | skilled, unskilled, high qualif, unemp | Socioeconomic class bias |

---

## Model Performance

Results on German Credit dataset after full pipeline:

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Random Forest | 73.0% | 0.745 |
| XGBoost | 73.0% | 0.754 |
| **Ensemble** | **78.0%** | **0.775** |

- RF Out-of-Bag Score: **0.826** (most reliable estimate on small dataset)
- Features after selection: **33 of 65** engineered features retained
- Class imbalance handled via **SMOTE** oversampling

---

## Fairness Thresholds

The system flags bias when:

| Metric | Acceptable | Flagged |
|---|---|---|
| Demographic Parity Difference | ≤ 0.20 | > 0.20 |
| Equalized Odds Difference | ≤ 0.20 | > 0.20 |

When bias is detected, the system automatically simulates **reweighting mitigation** and shows before/after comparison in the dashboard.

---

## Outputs

All reports are saved to the `reports/` directory:

| File | Description |
|---|---|
| `model_metrics.json` | Accuracy, AUC, classification report, optimal threshold |
| `explainability_summary.json` | Mean absolute SHAP values per feature |
| `fairness_summary.json` | Demographic parity and equalized odds differences |
| `dataset_profile.json` | Row/column counts, missingness, class distribution |
| `trajectory_summary.json` | PSI drift scores, bucket performance by sensitive group |
| `model_history.json` | Historical log of all runs with metrics over time |
| `model_predictions.csv` | Test set predictions with y_true, y_pred, sensitive column |
| `audit_report.json` | Combined audit bundle (all artifacts) |
| `audit_report.pdf` | Human-readable audit PDF for regulatory review |

---

## Key Design Decisions

**Why Ensemble over single model?**
The VotingClassifier combining RF and XGBoost consistently outperforms either model alone on this dataset, achieving 78% accuracy vs 73% individually.

**Why fairness-aware threshold instead of 0.5?**
Standard 0.5 threshold maximises accuracy but can amplify demographic bias. The system searches for a threshold that satisfies the 0.20 fairness limit while maximising accuracy.

**Why OOB score instead of CV accuracy?**
With only 1,000 samples, a single train/test split has high variance. The RF Out-of-Bag score (0.826) uses all training data and is a more reliable performance estimate.

**Why SMOTE on training data only?**
Oversampling is applied only to the training set before feature selection to prevent data leakage into evaluation.

---

## Optional — Notebooks

```powershell
jupyter notebook notebooks/
```

---

## Notes

- Numeric sensitive attributes (e.g. `age`) are automatically binned into groups for fairness analysis
- Feature selection reduces 65 engineered features to ~33, removing noise from polynomial and binned features
- The audit PDF is designed to be regulation-ready, referencing the 0.20 fairness threshold consistent with industry standards
- Trajectory monitoring requires at least 2 runs to show meaningful trends in the dashboard charts