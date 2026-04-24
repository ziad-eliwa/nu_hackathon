#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "artifacts/data/DeepX_hidden_test_predictions_baseline.csv"
JSON_PATH = ROOT / "artifacts/data/DeepX_hidden_test_predictions_baseline.json"

def main():
    df = pd.read_csv(CSV_PATH)
    if "review_id" not in df.columns or "pred_star_rating" not in df.columns:
        raise ValueError("Prediction CSV does not have required columns.")
    data = []
    for _, row in df.iterrows():
        data.append({"review_id": int(row["review_id"]), "pred_star_rating": int(row["pred_star_rating"])} )
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"JSON predictions written to {JSON_PATH}")

if __name__ == "__main__":
    main()
