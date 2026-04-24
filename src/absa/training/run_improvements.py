from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from absa.config.settings import DataPaths, InferenceSettings
from absa.data.io import load_labeled_reviews
from absa.data.schemas import PredictionRecord, ReviewInput
from absa.evaluation.metrics import evaluate_predictions
from absa.inference.predict import ABSAPredictor
from absa.inference.submission import write_submission_json
from absa.models.aspect_api import AspectEnsemblePredictor
from absa.models.sentiment_transformer import AspectConditionedSentimentModel
from absa.training.train_aspect import TransformerConfig, train_path_a
from absa.training.train_sentiment import train_and_evaluate_sentiment


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    normalize_profile: str
    aspect_word_features: int
    aspect_char_features: int
    aspect_alpha: float
    sentiment_word_features: int
    sentiment_char_features: int
    sentiment_c: float


def _evaluate_validation(validation_csv: str, artifacts_root: Path) -> dict:
    records = load_labeled_reviews(validation_csv)
    reviews = [
        ReviewInput(
            review_id=r.review_id,
            review_text=r.review_text,
            platform=r.platform,
            business_category=r.business_category,
        )
        for r in records
    ]

    aspect_model = AspectEnsemblePredictor.load_from_artifacts(artifacts_root)
    aspect_probs = aspect_model.predict_aspect_probs(records)

    class DictAspectProvider:
        def __init__(self, probs_dict):
            self._probs = probs_dict

        def predict_aspect_probs(self, batch):
            return [self._probs.get(r.review_id, {}) for r in batch]

    aspect_provider = DictAspectProvider(aspect_probs)
    sentiment_model = AspectConditionedSentimentModel.load(
        artifacts_root / "sentiment_model" / "sentiment_model.joblib"
    )
    settings = InferenceSettings.from_threshold_file(
        artifacts_root / "calibration" / "aspect_thresholds.json"
    )

    predictor = ABSAPredictor(
        sentiment_model=sentiment_model,
        aspect_provider=aspect_provider,
        settings=settings,
    )
    predictions = predictor.predict_reviews(reviews)

    reports_dir = artifacts_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    pred_json = reports_dir / "predictions_validation.json"
    write_submission_json(predictions, pred_json)

    gold = [
        PredictionRecord(
            review_id=r.review_id,
            aspects=list(r.aspects),
            aspect_sentiments=dict(r.aspect_sentiments),
        )
        for r in records
    ]
    return evaluate_predictions(gold, predictions)


def _run_one_experiment(
    cfg: ExperimentConfig,
    train_csv: str,
    validation_csv: str,
    root: Path,
) -> dict:
    artifact_dir = root / cfg.name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    os.environ["ABSA_NORMALIZE_PROFILE"] = cfg.normalize_profile

    start = time.perf_counter()
    aspect_result = train_path_a(
        train_csv=train_csv,
        validation_csv=validation_csv,
        artifacts_root=str(artifact_dir),
        transformer_config=TransformerConfig(
            max_word_features=cfg.aspect_word_features,
            max_char_features=cfg.aspect_char_features,
            alpha=cfg.aspect_alpha,
        ),
    )

    sentiment_result = train_and_evaluate_sentiment(
        train_csv=train_csv,
        validation_csv=validation_csv,
        output_dir=artifact_dir / "sentiment_model",
        model_kwargs={
            "word_max_features": cfg.sentiment_word_features,
            "char_max_features": cfg.sentiment_char_features,
            "c": cfg.sentiment_c,
        },
    )
    metrics = _evaluate_validation(validation_csv, artifact_dir)
    elapsed_sec = time.perf_counter() - start

    result = {
        "experiment": cfg.name,
        "normalize_profile": cfg.normalize_profile,
        "train_seconds": round(elapsed_sec, 2),
        "aspect_train": aspect_result,
        "sentiment_train": sentiment_result,
        "validation_metrics": metrics,
    }
    (artifact_dir / "reports" / "experiment_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def _summary_row(result: dict, baseline_tuple_f1: float | None) -> dict:
    metrics = result["validation_metrics"]
    tuple_f1 = float(metrics["tuple"]["f1"])
    return {
        "experiment": result["experiment"],
        "normalize_profile": result["normalize_profile"],
        "aspect_micro_f1": float(metrics["aspect_detection"]["micro_f1"]),
        "aspect_macro_f1": float(metrics["aspect_detection"]["macro_f1"]),
        "sentiment_macro_f1": float(metrics["sentiment_given_aspect"]["macro_f1"]),
        "tuple_f1": tuple_f1,
        "delta_tuple_f1": (tuple_f1 - baseline_tuple_f1) if baseline_tuple_f1 is not None else 0.0,
        "train_seconds": float(result["train_seconds"]),
    }


def _write_markdown_report(rows: list[dict], output_path: Path) -> None:
    lines = [
        "# Performance Improvement Results",
        "",
        "All experiments were run on the same train/validation split and evaluated with end-to-end tuple F1.",
        "",
        "| Experiment | Normalize | Aspect Micro F1 | Aspect Macro F1 | Sentiment Macro F1 | Tuple F1 | Delta Tuple F1 | Train Sec |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {experiment} | {normalize_profile} | {aspect_micro_f1:.4f} | {aspect_macro_f1:.4f} | "
            "{sentiment_macro_f1:.4f} | {tuple_f1:.4f} | {delta_tuple_f1:+.4f} | {train_seconds:.1f} |".format(**row)
        )

    best = max(rows, key=lambda r: r["tuple_f1"])
    lines.extend(
        [
            "",
            "## Best Configuration",
            "",
            f"- Experiment: {best['experiment']}",
            f"- Tuple F1: {best['tuple_f1']:.4f}",
            f"- Sentiment Macro F1: {best['sentiment_macro_f1']:.4f}",
            f"- Aspect Micro F1: {best['aspect_micro_f1']:.4f}",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_all(train_csv: str, validation_csv: str, output_root: Path) -> dict:
    experiments = [
        ExperimentConfig(
            name="exp_01_balanced_baseline",
            normalize_profile="balanced",
            aspect_word_features=40000,
            aspect_char_features=30000,
            aspect_alpha=5e-6,
            sentiment_word_features=40000,
            sentiment_char_features=30000,
            sentiment_c=2.0,
        ),
        ExperimentConfig(
            name="exp_02_aggressive_preprocess",
            normalize_profile="aggressive",
            aspect_word_features=40000,
            aspect_char_features=30000,
            aspect_alpha=5e-6,
            sentiment_word_features=40000,
            sentiment_char_features=30000,
            sentiment_c=2.0,
        ),
        ExperimentConfig(
            name="exp_03_aggressive_high_capacity",
            normalize_profile="aggressive",
            aspect_word_features=50000,
            aspect_char_features=40000,
            aspect_alpha=3e-6,
            sentiment_word_features=50000,
            sentiment_char_features=40000,
            sentiment_c=2.5,
        ),
        ExperimentConfig(
            name="exp_04_balanced_stronger_regularization",
            normalize_profile="balanced",
            aspect_word_features=45000,
            aspect_char_features=35000,
            aspect_alpha=4e-6,
            sentiment_word_features=45000,
            sentiment_char_features=35000,
            sentiment_c=3.0,
        ),
    ]

    output_root.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    summary_rows: list[dict] = []
    baseline_tuple_f1: float | None = None

    for exp in experiments:
        result = _run_one_experiment(exp, train_csv, validation_csv, output_root)
        all_results.append(result)

        if baseline_tuple_f1 is None:
            baseline_tuple_f1 = float(result["validation_metrics"]["tuple"]["f1"])

        summary_rows.append(_summary_row(result, baseline_tuple_f1))

    results_json = output_root / "all_experiment_results.json"
    results_json.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")

    _write_markdown_report(summary_rows, output_root / "results_report.md")
    return {
        "results_json": str(results_json),
        "results_markdown": str(output_root / "results_report.md"),
        "num_experiments": len(all_results),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = DataPaths()
    parser = argparse.ArgumentParser(description="Run performance improvement experiments")
    parser.add_argument("--train-csv", default=str(defaults.train_csv))
    parser.add_argument("--validation-csv", default=str(defaults.validation_csv))
    parser.add_argument("--output-root", default="artifacts/improvements")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    summary = run_all(
        train_csv=args.train_csv,
        validation_csv=args.validation_csv,
        output_root=Path(args.output_root),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
