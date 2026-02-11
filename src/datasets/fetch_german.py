from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1].parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.datasets import fetch_openml
from src.utils.data import preprocess, split_and_save


def main():
    print("Fetching 'credit-g' from OpenML...")
    ds = fetch_openml(name="credit-g", version=1, as_frame=True)
    df = ds.frame

    df = df.rename(columns={"class": "default"})

    out_raw = Path("data/raw")
    out_raw.mkdir(parents=True, exist_ok=True)
    raw_path = out_raw / "credit.csv"
    df.to_csv(raw_path, index=False)
    print(f"Raw dataset written to {raw_path}")

    df_p = preprocess(df)
    out_proc = Path("data/processed")
    train_path, test_path = split_and_save(df_p, str(out_proc), test_size=0.2, random_state=42)

    print(f"Processed train: {train_path}")
    print(f"Processed test : {test_path}")


if __name__ == "__main__":
    main()
