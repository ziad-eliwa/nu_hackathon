#!/usr/bin/env python3
"""
Baseline predictor for DeepX data using a simple TF-IDF + Logistic Regression
model trained on the DeepX_train.csv dataset.

This is a pragmatic fallback when the experiment 7 model cannot be loaded
in this environment. It provides deterministic predictions on
artifacts/data/DeepX_hidden_test.csv to unblock the workflow.
"""
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

ROOT = Path(__file__).resolve().parents[1]  # repository root
TRAIN_PATH = ROOT / "artifacts/data/DeepX_train.csv"
TEST_PATH = ROOT / "artifacts/data/DeepX_hidden_test.csv"
OUTPUT_PATH = ROOT / "artifacts/data/DeepX_hidden_test_predictions_baseline.csv"

def main():
    # Load training data
    df = pd.read_csv(TRAIN_PATH)
    # Expect columns: review_text, star_rating at minimum
    if "review_text" not in df.columns or "star_rating" not in df.columns:
        raise ValueError("Training data is missing required columns.")
    train_text = df["review_text"].astype(str).fillna("")
    train_y = df["star_rating"].astype(int)

    # Build a simple text classifier pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"))
    ])

    # Train on full training set
    pipeline.fit(train_text, train_y)

    # Load test data
    test_df = pd.read_csv(TEST_PATH)
    if "review_text" not in test_df.columns:
        raise ValueError("Test data missing 'review_text' column.")
    test_text = test_df["review_text"].astype(str).fillna("")
    # Predict
    preds = pipeline.predict(test_text)
    # Prepare output: keep the review_id for traceability if present
    out = pd.DataFrame({
        "review_id": test_df.get("review_id"),
        "pred_star_rating": preds,
    })
    # Save predictions
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Predictions written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
