"""Evaluation metrics for ABSA.

Metrics implemented per architecture plan:
- aspect detection micro/macro F1
- sentiment macro F1 (given predicted+gold overlap)
- end-to-end tuple F1 over (aspect, sentiment)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import classification_report, f1_score

from absa.config.taxonomy import ASPECT_TAXONOMY, SENTIMENT_LABELS
from absa.data.schemas import PredictionRecord


def _align_by_review_id(
    gold: Sequence[PredictionRecord],
    pred: Sequence[PredictionRecord],
) -> tuple[list[PredictionRecord], list[PredictionRecord]]:
    """Align two prediction lists by review_id to avoid order assumptions."""
    gold_map = {item.review_id: item for item in gold}
    pred_map = {item.review_id: item for item in pred}
    common_ids = sorted(set(gold_map).intersection(pred_map))

    if not common_ids:
        raise ValueError("No overlapping review_id values between gold and predicted")

    return [gold_map[idx] for idx in common_ids], [pred_map[idx] for idx in common_ids]


def _as_multilabel_matrix(records: Sequence[PredictionRecord]) -> np.ndarray:
    """Convert aspect lists to binary matrix for multilabel F1."""
    matrix = np.zeros((len(records), len(ASPECT_TAXONOMY)), dtype=int)
    aspect_to_idx = {aspect: idx for idx, aspect in enumerate(ASPECT_TAXONOMY)}

    for row_idx, record in enumerate(records):
        for aspect in record.aspects:
            col_idx = aspect_to_idx.get(aspect)
            if col_idx is not None:
                matrix[row_idx, col_idx] = 1
    return matrix


def compute_aspect_detection_f1(
    gold: Sequence[PredictionRecord],
    pred: Sequence[PredictionRecord],
) -> dict:
    """Compute micro/macro/per-aspect F1 for multi-label aspect detection."""
    aligned_gold, aligned_pred = _align_by_review_id(gold, pred)

    y_true = _as_multilabel_matrix(aligned_gold)
    y_pred = _as_multilabel_matrix(aligned_pred)

    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    per_aspect_f1: dict[str, float] = {}
    for idx, aspect in enumerate(ASPECT_TAXONOMY):
        per_aspect_f1[aspect] = float(
            f1_score(y_true[:, idx], y_pred[:, idx], zero_division=0)
        )

    return {
        "num_reviews": int(len(aligned_gold)),
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "per_aspect_f1": per_aspect_f1,
    }


def compute_sentiment_macro_f1_given_aspect(
    gold: Sequence[PredictionRecord],
    pred: Sequence[PredictionRecord],
) -> dict:
    """Compute sentiment macro F1 on overlapping predicted+gold aspects.

    We explicitly report coverage, because this metric ignores cases where an
    aspect was not predicted at all.
    """
    aligned_gold, aligned_pred = _align_by_review_id(gold, pred)

    y_true: list[str] = []
    y_pred: list[str] = []
    total_gold_aspect_instances = 0

    for gold_item, pred_item in zip(aligned_gold, aligned_pred):
        total_gold_aspect_instances += len(gold_item.aspects)

        pred_map = pred_item.aspect_sentiments
        for aspect in gold_item.aspects:
            if aspect in pred_map:
                y_true.append(gold_item.aspect_sentiments[aspect])
                y_pred.append(pred_map[aspect])

    if not y_true:
        return {
            "num_overlap_aspect_instances": 0,
            "coverage_over_gold_aspects": 0.0,
            "macro_f1": 0.0,
            "report": {},
        }

    macro_f1 = float(
        f1_score(
            y_true,
            y_pred,
            labels=list(SENTIMENT_LABELS),
            average="macro",
            zero_division=0,
        )
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=list(SENTIMENT_LABELS),
        output_dict=True,
        zero_division=0,
    )

    coverage = (
        float(len(y_true) / total_gold_aspect_instances)
        if total_gold_aspect_instances > 0
        else 0.0
    )

    return {
        "num_overlap_aspect_instances": int(len(y_true)),
        "coverage_over_gold_aspects": coverage,
        "macro_f1": macro_f1,
        "report": report,
    }


def compute_tuple_f1(
    gold: Sequence[PredictionRecord],
    pred: Sequence[PredictionRecord],
) -> dict:
    """Compute end-to-end tuple precision/recall/F1 over (aspect, sentiment)."""
    aligned_gold, aligned_pred = _align_by_review_id(gold, pred)

    true_positive = 0
    false_positive = 0
    false_negative = 0

    for gold_item, pred_item in zip(aligned_gold, aligned_pred):
        gold_tuples = {(aspect, gold_item.aspect_sentiments[aspect]) for aspect in gold_item.aspects}
        pred_tuples = {(aspect, pred_item.aspect_sentiments[aspect]) for aspect in pred_item.aspects}

        true_positive += len(gold_tuples.intersection(pred_tuples))
        false_positive += len(pred_tuples - gold_tuples)
        false_negative += len(gold_tuples - pred_tuples)

    precision = (
        float(true_positive / (true_positive + false_positive))
        if (true_positive + false_positive) > 0
        else 0.0
    )
    recall = (
        float(true_positive / (true_positive + false_negative))
        if (true_positive + false_negative) > 0
        else 0.0
    )
    f1 = (
        float((2 * precision * recall) / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "num_reviews": int(len(aligned_gold)),
        "true_positive": int(true_positive),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_predictions(
    gold: Sequence[PredictionRecord],
    pred: Sequence[PredictionRecord],
) -> dict:
    """Run full metric bundle in one helper call."""
    return {
        "aspect_detection": compute_aspect_detection_f1(gold, pred),
        "sentiment_given_aspect": compute_sentiment_macro_f1_given_aspect(gold, pred),
        "tuple": compute_tuple_f1(gold, pred),
    }
