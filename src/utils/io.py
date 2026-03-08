from pathlib import Path
import json


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def save_csv(df, path):
    path = Path(path)  # ✅ Fixed: Path was used without importing it
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)