"""Train the Path B aspect-conditioned sentiment model.

This trainer intentionally keeps each step explicit and commented so teammates
can quickly inspect assumptions and replace components later.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, f1_score

from absa.models.sentiment_transformer import (
    AspectConditionedSentimentModel,
    build_training_examples_from_dataframe,
)


def _load_labeled_csv(path: str | Path) -> pd.DataFrame:
    """Load and lightly validate expected columns from labeled CSV."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Labeled CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {"review_id", "review_text", "aspect_sentiments"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing)} in {csv_path}"
        )
    return df


def _evaluate_model(model: AspectConditionedSentimentModel, df: pd.DataFrame) -> dict:
    """Evaluate sentiment model on one labeled dataframe split."""
    texts, labels, review_ids, aspects = build_training_examples_from_dataframe(df)

    if not texts:
        return {
            "num_samples": 0,
            "macro_f1": 0.0,
            "report": {},
            "note": "No sentiment samples found in this split",
        }

    predictions = model.predict_conditioned_texts(texts)
    macro_f1 = float(f1_score(labels, predictions, average="macro", zero_division=0))
    report = classification_report(
        labels,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    return {
        "num_samples": len(texts),
        "num_unique_reviews": len(set(review_ids)),
        "num_unique_aspects": len(set(aspects)),
        "macro_f1": macro_f1,
        "report": report,
    }


def train_and_evaluate_sentiment(
    train_csv: str | Path,
    validation_csv: str | Path,
    output_dir: str | Path,
) -> dict:
    """Train model and export artifacts/metrics.

    Returns:
        Dictionary summary that can be printed or logged by callers.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: load both labeled splits.
    train_df = _load_labeled_csv(train_csv)
    validation_df = _load_labeled_csv(validation_csv)

    # Step 2: build and train the model on train split.
    model = AspectConditionedSentimentModel()
    train_texts, train_labels, _, _ = build_training_examples_from_dataframe(train_df)
    model.fit(train_texts, train_labels)

    # Step 3: evaluate on train/validation for quick overfit and generalization checks.
    train_metrics = _evaluate_model(model, train_df)
    validation_metrics = _evaluate_model(model, validation_df)

    # Step 4: save binary model artifact and machine-readable metrics report.
    model_path = output_path / "sentiment_model.joblib"
    metrics_path = output_path / "sentiment_metrics.json"
    model.save(model_path)

    summary = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "train_csv": str(Path(train_csv)),
        "validation_csv": str(Path(validation_csv)),
        "model_path": str(model_path),
        "train": train_metrics,
        "validation": validation_metrics,
    }
    metrics_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    """Define CLI arguments for standalone training runs."""
    parser = argparse.ArgumentParser(
        description="Train Path B aspect-conditioned sentiment model"
    )
    parser.add_argument(
        "--train-csv",
        default="data/DeepX_train.csv",
        help="Path to labeled train CSV",
    )
    parser.add_argument(
        "--validation-csv",
        default="data/DeepX_validation.csv",
        help="Path to labeled validation CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/sentiment_model",
        help="Directory where model and metrics will be written",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint used by main.py or direct module execution."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    summary = train_and_evaluate_sentiment(
        train_csv=args.train_csv,
        validation_csv=args.validation_csv,
        output_dir=args.output_dir,
    )

    print("Sentiment training finished")
    print(f"Model artifact: {summary['model_path']}")
    print(f"Train macro-F1: {summary['train']['macro_f1']:.4f}")
    print(f"Validation macro-F1: {summary['validation']['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
