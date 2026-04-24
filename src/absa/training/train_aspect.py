from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from absa.config.settings import ArtifactPaths, DataPaths
from absa.config.taxonomy import ASPECTS
from absa.data.io import load_labeled_reviews
from absa.data.schemas import LabeledReviewRecord
from absa.models.aspect_api import AspectEnsemblePredictor
from absa.models.aspect_linear import AspectLinearModel
from absa.models.aspect_transformer import AspectTransformerModel, TransformerConfig
from absa.training.calibrate import apply_thresholds, optimize_aspect_thresholds, save_thresholds_config


def labels_matrix(records: list[LabeledReviewRecord]) -> np.ndarray:
    matrix = np.zeros((len(records), len(ASPECTS)), dtype=np.int32)
    aspect_to_idx = {aspect: idx for idx, aspect in enumerate(ASPECTS)}
    for row_idx, record in enumerate(records):
        for aspect in record.aspects:
            matrix[row_idx, aspect_to_idx[aspect]] = 1
    return matrix


def _find_best_blend_weight(
    y_true: np.ndarray,
    linear_probs: np.ndarray,
    transformer_probs: np.ndarray,
) -> float:
    best_weight = 0.65
    best_score = -1.0
    for weight in np.linspace(0.0, 1.0, 9):
        blended = weight * linear_probs + (1.0 - weight) * transformer_probs
        score = f1_score(y_true, (blended >= 0.5).astype(int), average="micro", zero_division=0)
        if score > best_score:
            best_score = score
            best_weight = float(weight)
    return best_weight


def _length_bucket(text: str) -> str:
    length = len(text)
    if length < 60:
        return "short"
    if length < 200:
        return "medium"
    return "long"


def _count_bucket(aspects_count: int) -> str:
    if aspects_count == 1:
        return "1"
    if aspects_count == 2:
        return "2"
    return "3+"


def _slice_metrics(
    records: list[LabeledReviewRecord],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}

    def _compute(indices: list[int]) -> dict[str, float]:
        if not indices:
            return {"micro_f1": 0.0, "macro_f1": 0.0, "count": 0}
        return {
            "micro_f1": float(
                f1_score(y_true[indices], y_pred[indices], average="micro", zero_division=0)
            ),
            "macro_f1": float(
                f1_score(y_true[indices], y_pred[indices], average="macro", zero_division=0)
            ),
            "count": len(indices),
        }

    by_platform: dict[str, list[int]] = {}
    by_length: dict[str, list[int]] = {}
    by_aspect_count: dict[str, list[int]] = {}
    for idx, record in enumerate(records):
        by_platform.setdefault(record.platform, []).append(idx)
        by_length.setdefault(_length_bucket(record.review_text), []).append(idx)
        by_aspect_count.setdefault(_count_bucket(len(record.aspects)), []).append(idx)

    result["platform"] = {k: _compute(v) for k, v in by_platform.items()}
    result["length_bucket"] = {k: _compute(v) for k, v in by_length.items()}
    result["aspects_per_review"] = {k: _compute(v) for k, v in by_aspect_count.items()}
    return result


def train_path_a(
    train_csv: str,
    validation_csv: str,
    artifacts_root: str,
) -> dict[str, object]:
    train_records = load_labeled_reviews(train_csv)
    val_records = load_labeled_reviews(validation_csv)
    y_train = labels_matrix(train_records)
    y_val = labels_matrix(val_records)

    linear_model = AspectLinearModel()
    linear_model.fit(train_records, y_train)
    linear_probs = linear_model.predict_proba(val_records)

    transformer_config = TransformerConfig()
    transformer_model = AspectTransformerModel(config=transformer_config)
    transformer_model.fit(train_records, y_train)
    transformer_probs = transformer_model.predict_proba(val_records)

    best_weight = _find_best_blend_weight(y_val, linear_probs, transformer_probs)
    blended_probs = best_weight * linear_probs + (1.0 - best_weight) * transformer_probs
    thresholds = optimize_aspect_thresholds(y_true=y_val, y_prob=blended_probs)

    y_pred = np.zeros_like(y_val)
    for idx, record in enumerate(val_records):
        prob_map = {aspect: float(blended_probs[idx, aspect_idx]) for aspect_idx, aspect in enumerate(ASPECTS)}
        selected = apply_thresholds(prob_map=prob_map, thresholds=thresholds)
        for aspect in selected:
            y_pred[idx, ASPECTS.index(aspect)] = 1

    eval_payload: dict[str, object] = {
        "metrics": {
            "micro_f1": float(f1_score(y_val, y_pred, average="micro", zero_division=0)),
            "macro_f1": float(f1_score(y_val, y_pred, average="macro", zero_division=0)),
        },
        "slice_metrics": _slice_metrics(val_records, y_val, y_pred),
        "ensemble": {
            "linear_weight": best_weight,
            "transformer_weight": float(1.0 - best_weight),
        },
    }

    root = Path(artifacts_root)
    model_dir = root / "aspect_model"
    calibration_dir = root / "calibration"
    reports_dir = root / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    calibration_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    linear_model.save(model_dir / "aspect_linear.pkl")
    transformer_model.save(model_dir / "aspect_transformer.pt")
    AspectEnsemblePredictor(
        linear_model=linear_model,
        transformer_model=transformer_model,
        linear_weight=best_weight,
    ).save_config(model_dir / "ensemble_config.json")

    save_thresholds_config(
        path=calibration_dir / "aspect_thresholds.json",
        thresholds=thresholds,
    )

    (reports_dir / "aspect_eval.json").write_text(
        json.dumps(eval_payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    predictor = AspectEnsemblePredictor(
        linear_model=linear_model,
        transformer_model=transformer_model,
        linear_weight=best_weight,
    )
    sampled = val_records[:50]
    sample_probs = predictor.predict_aspect_probs(sampled)
    (reports_dir / "aspect_sample_predictions_50.json").write_text(
        json.dumps(sample_probs, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return eval_payload


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = DataPaths()
    artifacts = ArtifactPaths()
    parser = argparse.ArgumentParser(description="Train Path A aspect models and calibration.")
    parser.add_argument("--train-csv", default=str(defaults.train_csv))
    parser.add_argument("--validation-csv", default=str(defaults.validation_csv))
    parser.add_argument("--artifacts-root", default=str(artifacts.root))
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    payload = train_path_a(
        train_csv=args.train_csv,
        validation_csv=args.validation_csv,
        artifacts_root=args.artifacts_root,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
