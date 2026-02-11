from pathlib import Path
import joblib
import pandas as pd


def main(model_pkl: str = 'artifacts/model_xgb.pkl', test_parquet: str = 'data/processed/test.parquet', out_csv: str = 'reports/model_predictions.csv'):
    model = joblib.load(model_pkl)

    df_test = pd.read_parquet(test_parquet)

    target = None
    for c in ['default', 'target', 'y', 'label', 'class', 'status']:
        if c in df_test.columns:
            target = c
            break

    X = df_test.copy()
    if target is not None:
        y = X[target]
        X = X.drop(columns=[target])
    else:
        y = None

    for id_col in ('Unnamed: 0', 'id', 'ID', 'Id'):
        if id_col in X.columns:
            X = X.drop(columns=[id_col])

    for col in X.select_dtypes(include=['number']).columns:
        X[col] = X[col].fillna(X[col].median())
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = X[col].fillna('__missing__')

    X_enc = pd.get_dummies(X, drop_first=True)

    try:
        preds = model.predict(X_enc)
    except Exception:
        import numpy as np
        model_features = model.get_booster().feature_names
        for f in model_features:
            if f not in X_enc.columns:
                X_enc[f] = 0
        X_enc = X_enc[model_features]
        preds = model.predict(X_enc)

    out = X.copy()
    if y is not None:
        out['y_true'] = list(y)
    out['y_pred'] = list(preds)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote predictions to {out_path}")


if __name__ == '__main__':
    main()
