from __future__ import annotations

import numpy as np

from absa.training.calibrate import apply_thresholds, optimize_aspect_thresholds


def test_optimize_aspect_thresholds_returns_all_aspects():
    y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    y_prob = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8]])
    thresholds = optimize_aspect_thresholds(y_true, y_prob, aspects=("food", "service"))
    assert set(thresholds) == {"food", "service"}
    assert all(0.1 <= value <= 0.9 for value in thresholds.values())


def test_apply_thresholds_fallback_none():
    selected = apply_thresholds(
        prob_map={"food": 0.1, "general": 0.2, "none": 0.8},
        thresholds={"food": 0.5, "general": 0.5, "none": 0.9},
        fallback_none_threshold=0.7,
        fallback_general_threshold=0.8,
    )
    assert selected == ["none"]

