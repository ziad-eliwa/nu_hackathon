from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from absa.config.taxonomy import ASPECT_TAXONOMY as ASPECTS, ordered_aspects
from absa.config.settings import DEFAULT_NONE_THRESHOLD, DEFAULT_GENERAL_THRESHOLD


def optimize_aspect_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    aspects: tuple[str, ...] = ASPECTS,
    none_focus: bool = True,
) -> dict[str, float]:
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    thresholds: dict[str, float] = {}
    candidates = np.linspace(0.1, 0.9, 17)
    
    none_idx = aspects.index("none") if "none" in aspects else None
    
    for idx, aspect in enumerate(aspects):
        best_score = -1.0
        best_threshold = 0.5
        
        aspect_true = y_true[:, idx]
        aspect_has_positive = aspect_true.sum() > 0
        
        if not aspect_has_positive:
            thresholds[aspect] = 0.5
            continue
        
        if none_focus and aspect == "none" and none_idx is not None:
            none_candidates = np.linspace(0.05, 0.5, 10)
            for threshold in none_candidates:
                pred = (y_prob[:, idx] >= threshold).astype(int)
                score = f1_score(y_true[:, idx], pred, zero_division=0)
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
        else:
            for threshold in candidates:
                pred = (y_prob[:, idx] >= threshold).astype(int)
                score = f1_score(y_true[:, idx], pred, zero_division=0)
                if score > best_score or (score == best_score and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
                    best_score = score
                    best_threshold = threshold
        
        thresholds[aspect] = float(best_threshold)
    
    if none_focus and none_idx is not None:
        thresholds["none"] = max(0.15, thresholds.get("none", 0.5) * 0.8)
    
    return thresholds


def apply_thresholds(
    prob_map: dict[str, float],
    thresholds: dict[str, float],
    fallback_none_threshold: float = DEFAULT_NONE_THRESHOLD,
    fallback_general_threshold: float = DEFAULT_GENERAL_THRESHOLD,
    none_aggressive: bool = True,
) -> list[str]:
    selected = [aspect for aspect, prob in prob_map.items() if prob >= thresholds.get(aspect, 0.5)]
    
    if none_aggressive:
        none_prob = prob_map.get("none", 0.0)
        concrete_probs = {a: prob_map.get(a, 0.0) for a in ASPECTS if a != "none"}
        max_concrete_prob = max(concrete_probs.values()) if concrete_probs else 0.0
        
        if none_prob > max_concrete_prob * 1.2:
            return ["none"]
    
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
    fallback_none_threshold: float = DEFAULT_NONE_THRESHOLD,
    fallback_general_threshold: float = DEFAULT_GENERAL_THRESHOLD,
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

