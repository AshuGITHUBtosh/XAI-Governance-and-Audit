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


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for id_col in ("id", "ID", "Id"):
        if id_col in df.columns:
            df = df.drop(columns=[id_col])

    df.columns = [c.strip() for c in df.columns]

    target = _detect_target(df)
    if target is not None:
        if df[target].dtype == object:
            df[target] = df[target].str.strip().str.lower().map({"yes": 1, "no": 0, "good": 0, "bad": 1})
        df[target] = pd.to_numeric(df[target], errors="coerce")
    return df


def split_and_save(df: pd.DataFrame, out_dir: str, test_size: float = 0.2, random_state: int = 42):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target = _detect_target(df)
    if target is not None:
        X = df.drop(columns=[target])
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() > 1 else None
        )
        train = X_train.copy()
        train[target] = y_train
        test = X_test.copy()
        test[target] = y_test
    else:
        train, test = train_test_split(df, test_size=test_size, random_state=random_state)

    train_path = out_dir / "train.parquet"
    test_path = out_dir / "test.parquet"
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)

    return str(train_path), str(test_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ingest raw CSV and produce train/test parquet files")
    parser.add_argument("--raw", type=str, default="data/raw/credit.csv", help="Path to raw CSV file")
    parser.add_argument("--out", type=str, default="data/processed", help="Output folder for parquet files")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading raw data from {args.raw}")
    df = load_raw(args.raw)
    print(f"Raw data shape: {df.shape}")

    df = preprocess(df)
    print(f"After preprocessing shape: {df.shape}")

    train_path, test_path = split_and_save(df, args.out, test_size=args.test_size, random_state=args.seed)
    print(f"Wrote train -> {train_path}")
    print(f"Wrote test  -> {test_path}")


if __name__ == "__main__":
    main()
