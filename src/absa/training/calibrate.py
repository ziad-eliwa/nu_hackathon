from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from absa.config.taxonomy import ASPECT_TAXONOMY as ASPECTS, ordered_aspects


def optimize_aspect_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    aspects: tuple[str, ...] = ASPECTS,
) -> dict[str, float]:
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    thresholds: dict[str, float] = {}
    candidates = np.linspace(0.1, 0.9, 17)
    for idx, aspect in enumerate(aspects):
        best_score = -1.0
        best_threshold = 0.5
        for threshold in candidates:
            pred = (y_prob[:, idx] >= threshold).astype(int)
            score = f1_score(y_true[:, idx], pred, zero_division=0)
            if score > best_score or (score == best_score and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
                best_score = score
                best_threshold = threshold
        thresholds[aspect] = float(best_threshold)
    return thresholds


def apply_thresholds(
    prob_map: dict[str, float],
    thresholds: dict[str, float],
    fallback_none_threshold: float = 0.6,
    fallback_general_threshold: float = 0.55,
) -> list[str]:
    selected = [aspect for aspect, prob in prob_map.items() if prob >= thresholds.get(aspect, 0.5)]
    if "none" in selected and len(selected) > 1:
        selected = [aspect for aspect in selected if aspect != "none"]
    if selected:
        return ordered_aspects(selected)
    none_prob = prob_map.get("none", 0.0)
    general_prob = prob_map.get("general", 0.0)
    if none_prob >= fallback_none_threshold:
        return ["none"]
    if general_prob >= fallback_general_threshold:
        return ["general"]
    best_aspect = max(prob_map.items(), key=lambda item: item[1])[0]
    return [best_aspect]


def save_thresholds_config(
    path: str | Path,
    thresholds: dict[str, float],
    fallback_none_threshold: float = 0.6,
    fallback_general_threshold: float = 0.55,
    version: str = "path-a-v1",
) -> None:
    payload = {
        "version": version,
        "created_at": datetime.now(UTC).isoformat(),
        "thresholds": thresholds,
        "fallback_policy": {
            "none_threshold": fallback_none_threshold,
            "general_threshold": fallback_general_threshold,
        },
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

