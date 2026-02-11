# explainable-ai-finance

Minimal project scaffold for explainable AI in finance.

Structure created for data, src, reports, dashboards, and notebooks.

## Setup Commands

```powershell
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
```

## Data & Model Pipeline

```powershell
# 3. Fetch and preprocess the German Credit dataset
python src/datasets/fetch_german.py

# 4. Train the XGBoost model
python src/model_training.py --train data/processed/train.parquet --test data/processed/test.parquet --target default --model artifacts/model_xgb.json

# 5. Generate dataset profile
python src/utils/profile.py --train data/processed/train.parquet --out reports/dataset_profile.json

# 6. Generate explainability summary (SHAP values)
python src/explainability_layer.py --model artifacts/model_xgb.pkl --data data/processed/test.parquet --out reports/explainability_summary.json

# 7. Compute fairness metrics
python src/metrics/fairness.py --preds reports/model_predictions.csv --sensitive sex --y_true y_true --y_pred y_pred --out reports/fairness_summary.json

# 8. Generate audit report (combines all summaries)
python src/utils/evidence_pack.py
```

## Optional - Notebooks & Dashboard

```powershell
# 9. Start Jupyter for exploration
jupyter notebook notebooks/

# 10. Launch Streamlit dashboard (if configured)
streamlit run src/dashboard.py
```

**Note:** Commands 3-4 are essential to start. Commands 5-8 generate reports and can run in sequence or independently once the model is trained.
