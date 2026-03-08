from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")
    return pd.read_csv(path)


def _detect_target(df: pd.DataFrame) -> Optional[str]:
    candidates = ["default", "target", "y", "label", "class", "status"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _encode_target(series: pd.Series) -> pd.Series:
    """
    Robustly encode target column to binary 0/1.

    Handles all common formats found in credit datasets:
      - 'good'/'bad'       → 0/1
      - 'yes'/'no'         → 1/0
      - '1'/'2' or 1/2     → 0/1  (German Credit integer coding)
      - '0'/'1' or 0/1     → kept as-is
      - True/False         → 1/0
    """
    s = series.copy()

    # --- String-based encoding ---
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        s = s.astype(str).str.strip().str.lower()

        # Map all known string labels
        label_map = {
            "good": 0, "bad": 1,
            "yes": 1,  "no": 0,
            "true": 1, "false": 0,
            "1": 1,    "0": 0,
            "2": 1,    # German Credit: 1=good, 2=bad
        }
        mapped = s.map(label_map)

        # If mapping left NaNs, check if values are purely numeric strings
        if mapped.isna().any():
            unmapped = s[mapped.isna()].unique()
            print(f"  Warning: unmapped target values {unmapped} — attempting numeric fallback")
            # Try numeric conversion for anything not in map
            numeric_fallback = pd.to_numeric(s, errors="coerce")
            # Fill NaN slots with numeric fallback
            mapped = mapped.fillna(numeric_fallback)

        s = mapped

    # --- Integer/numeric encoding ---
    else:
        s = pd.to_numeric(s, errors="coerce")

        unique_vals = sorted(s.dropna().unique())

        # German Credit integer encoding: 1=good(0), 2=bad(1)
        if set(unique_vals).issubset({1, 2}):
            print(f"  Detected 1/2 integer encoding — mapping 1→0 (good), 2→1 (bad)")
            s = s.map({1: 0, 2: 1})

        # Already 0/1 — leave as-is
        elif set(unique_vals).issubset({0, 1}):
            pass

        # Unknown numeric range — normalise to 0/1 by ranking
        else:
            print(f"  Unknown numeric target range {unique_vals} — binarising at median")
            median = s.median()
            s = (s > median).astype(float)

    s = s.astype(float)

    null_count = s.isna().sum()
    if null_count > 0:
        print(f"  Warning: {null_count} NaN values remain in target after encoding — these rows will be dropped during training")

    return s


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop id-like columns
    for id_col in ("id", "ID", "Id"):
        if id_col in df.columns:
            df = df.drop(columns=[id_col])

    df.columns = [c.strip() for c in df.columns]

    target = _detect_target(df)
    if target is not None:
        print(f"  Encoding target column '{target}' — sample values: {df[target].unique()[:5].tolist()}")
        df[target] = _encode_target(df[target])
        print(f"  After encoding — unique values: {sorted(df[target].dropna().unique().tolist())}")
        print(f"  Class distribution:\n{df[target].value_counts().to_string()}")

    return df


def split_and_save(df: pd.DataFrame, out_dir: str, test_size: float = 0.2, random_state: int = 42):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target = _detect_target(df)
    if target is not None:
        # Drop rows where target is still NaN
        before = len(df)
        df = df[df[target].notna()].copy()
        dropped = before - len(df)
        if dropped > 0:
            print(f"  Dropped {dropped} rows with NaN target")

        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if y.nunique() > 1 else None,
        )
        train = X_train.copy()
        train[target] = y_train
        test = X_test.copy()
        test[target] = y_test
    else:
        train, test = train_test_split(df, test_size=test_size, random_state=random_state)

    train_path = out_dir / "train.parquet"
    test_path  = out_dir / "test.parquet"
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path,  index=False)

    print(f"  Train: {len(train)} rows | Test: {len(test)} rows")
    return str(train_path), str(test_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ingest raw CSV and produce train/test parquet files")
    parser.add_argument("--raw",       type=str,   default="data/raw/credit.csv")
    parser.add_argument("--out",       type=str,   default="data/processed")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    print(f"Loading raw data from {args.raw}")
    df = load_raw(args.raw)
    print(f"Raw shape: {df.shape}")

    df = preprocess(df)
    print(f"After preprocessing shape: {df.shape}")

    train_path, test_path = split_and_save(
        df, args.out, test_size=args.test_size, random_state=args.seed
    )
    print(f"Wrote train → {train_path}")
    print(f"Wrote test  → {test_path}")


if __name__ == "__main__":
    main()