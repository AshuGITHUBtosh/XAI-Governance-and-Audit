from pathlib import Path
import json
from typing import List

import pandas as pd


def detect_target(df: pd.DataFrame) -> str:
    for c in ["default", "target", "y", "label", "class", "status"]:
        if c in df.columns:
            return c
    return None


def find_sensitive(df: pd.DataFrame) -> List[str]:
    candidates = ["sex", "gender", "age", "race", "ethnicity", "marital", "education"]
    found = [c for c in candidates if c in df.columns]
    return found


def profile(df: pd.DataFrame) -> dict:
    prof = {}
    prof["rows"] = int(df.shape[0])
    prof["columns"] = int(df.shape[1])

    target = detect_target(df)
    prof["target"] = target
    if target:
        vc = df[target].value_counts(dropna=False)
        prof["target_counts"] = vc.to_dict()

    prof["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
    miss = (df.isna().mean() * 100).round(3).sort_values(ascending=False)
    prof["missingness_percent_top"] = miss[miss > 0].head(20).to_dict()

    num = df.select_dtypes(include=["number"]) 
    if not num.empty:
        desc = num.describe().T
        prof["numeric_summary"] = desc[ ["count","mean","std","min","25%","50%","75%","max"] ].to_dict(orient="index")
        corr = num.corr().abs()
        corr_vals = []
        for i, col in enumerate(corr.columns):
            for j, col2 in enumerate(corr.columns):
                if j <= i:
                    continue
                corr_vals.append((col, col2, float(corr.iloc[i, j])))
        corr_vals = sorted(corr_vals, key=lambda x: x[2], reverse=True)[:20]
        prof["top_abs_correlations"] = [ {"col1": a, "col2": b, "abs_corr": v} for a,b,v in corr_vals ]

    prof["suggested_sensitive_columns"] = find_sensitive(df)

    return prof


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Profile dataset and write JSON report")
    parser.add_argument("--train", type=str, default="data/processed/train.parquet")
    parser.add_argument("--out", type=str, default="reports/dataset_profile.json")
    args = parser.parse_args()

    train_path = Path(args.train)
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")

    df = pd.read_parquet(train_path)
    prof = profile(df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(prof, f, indent=2)

    print(f"Wrote dataset profile to {out_path}")


if __name__ == "__main__":
    main()
