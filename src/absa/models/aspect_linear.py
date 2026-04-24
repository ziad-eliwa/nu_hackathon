from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from absa.config.taxonomy import ASPECT_TAXONOMY as ASPECTS
from absa.data.schemas import ReviewRecord
from absa.features.tfidf import TfidfFeatureExtractor
from absa.preprocess.metadata import record_to_metadata
from absa.preprocess.normalize import normalize_text


class AspectLinearModel:
    def __init__(self, feature_extractor: TfidfFeatureExtractor | None = None) -> None:
        self.feature_extractor = feature_extractor or TfidfFeatureExtractor()
        self.classifier = OneVsRestClassifier(
            LogisticRegression(
                max_iter=700,
                solver="liblinear",
                class_weight="balanced",
            )
        )
        self._is_fitted = False

    def fit(self, records: Sequence[ReviewRecord], labels: np.ndarray) -> None:
        texts = [normalize_text(record.review_text) for record in records]
        metadata = [record_to_metadata(record) for record in records]
        features = self.feature_extractor.fit_transform(texts, metadata)
        self.classifier.fit(features, labels)
        self._is_fitted = True

    def predict_proba(self, records: Sequence[ReviewRecord]) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted")
        texts = [normalize_text(record.review_text) for record in records]
        metadata = [record_to_metadata(record) for record in records]
        features = self.feature_extractor.transform(texts, metadata)
        probas = self.classifier.predict_proba(features)
        return np.asarray(probas)

    def predict_aspect_probs(self, review_batch: Sequence[ReviewRecord]) -> dict[str, dict[str, float]]:
        prob_matrix = self.predict_proba(review_batch)
        output: dict[str, dict[str, float]] = {}
        for idx, review in enumerate(review_batch):
            output[review.review_id] = {
                aspect: float(prob_matrix[idx, aspect_idx])
                for aspect_idx, aspect in enumerate(ASPECTS)
            }
        return output

    def save(self, model_path: str | Path) -> None:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, model_path: str | Path) -> "AspectLinearModel":
        with Path(model_path).open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError("Loaded model has unexpected type")
        model._is_fitted = True
        return model

