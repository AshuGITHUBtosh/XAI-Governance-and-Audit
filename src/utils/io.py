def save_json(obj, path):
    with open(path, 'w') as f:
        import json
        json.dump(obj, f, indent=2)


def save_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
