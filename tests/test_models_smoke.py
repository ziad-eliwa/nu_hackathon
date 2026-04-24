from __future__ import annotations

import numpy as np

from absa.config.taxonomy import ASPECTS
from absa.data.schemas import ReviewRecord
from absa.models.aspect_linear import AspectLinearModel
from absa.models.aspect_transformer import AspectTransformerModel, TransformerConfig


def _records() -> list[ReviewRecord]:
    return [
        ReviewRecord("1", "الخدمة ممتازة", 5, "d", "b", "c", "google_maps"),
        ReviewRecord("2", "الخدمة سيئة", 1, "d", "b", "c", "google_maps"),
        ReviewRecord("3", "الاكل رائع", 5, "d", "b", "c", "google_maps"),
        ReviewRecord("4", "الاسعار غالية", 1, "d", "b", "c", "google_maps"),
    ]


def _labels() -> np.ndarray:
    matrix = np.zeros((4, len(ASPECTS)), dtype=np.int32)
    matrix[0, ASPECTS.index("service")] = 1
    matrix[1, ASPECTS.index("service")] = 1
    matrix[2, ASPECTS.index("food")] = 1
    matrix[3, ASPECTS.index("price")] = 1
    return matrix


def test_linear_model_smoke():
    records = _records()
    labels = _labels()
    model = AspectLinearModel()
    model.fit(records, labels)
    probs = model.predict_proba(records)
    assert probs.shape == (len(records), len(ASPECTS))


def test_transformer_model_smoke():
    records = _records()
    labels = _labels()
    model = AspectTransformerModel(
        config=TransformerConfig(
            max_features=2_000,
            alpha=1e-4,
        )
    )
    model.fit(records, labels)
    probs = model.predict_proba(records)
    assert probs.shape == (len(records), len(ASPECTS))
