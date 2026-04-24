from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier

from absa.config.taxonomy import ASPECT_TAXONOMY as ASPECTS
from absa.data.schemas import ReviewRecord
from absa.preprocess.metadata import record_to_metadata
from absa.preprocess.normalize import normalize_text


@dataclass
class TransformerConfig:
    max_features: int | None = None
    max_word_features: int = 40_000
    max_char_features: int = 30_000
    alpha: float = 5e-6
    random_seed: int = 42


class AspectTransformerModel:
    """
    A lightweight transformer-surrogate for Path A.

    It uses richer token n-gram features and a linear neural-style classifier
    to provide a second, complementary probability channel to the sparse baseline.
    """

    def __init__(self, config: TransformerConfig | None = None) -> None:
        self.config = config or TransformerConfig()
        max_word_features = self.config.max_word_features
        max_char_features = self.config.max_char_features
        if self.config.max_features is not None:
            max_word_features = int(self.config.max_features)
            max_char_features = int(self.config.max_features)
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            max_features=max_word_features,
            min_df=2,
            lowercase=False,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 6),
            max_features=max_char_features,
            min_df=2,
            lowercase=False,
        )
        self.metadata_vectorizer = DictVectorizer(sparse=True)
        self.classifier = OneVsRestClassifier(
            SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=self.config.alpha,
                max_iter=2000,
                class_weight="balanced",
                random_state=self.config.random_seed,
            )
        )
        self._is_fitted = False

    def _build_features(self, records: Sequence[ReviewRecord], fit: bool):
        texts = [normalize_text(record.review_text) for record in records]
        metadata = [record_to_metadata(record) for record in records]
        if fit:
            word_features = self.word_vectorizer.fit_transform(texts)
            char_features = self.char_vectorizer.fit_transform(texts)
            meta_features = self.metadata_vectorizer.fit_transform(metadata)
        else:
            word_features = self.word_vectorizer.transform(texts)
            char_features = self.char_vectorizer.transform(texts)
            meta_features = self.metadata_vectorizer.transform(metadata)
        return hstack([word_features, char_features, meta_features]).tocsr()

    def fit(self, records: Sequence[ReviewRecord], labels: np.ndarray) -> None:
        features = self._build_features(records, fit=True)
        self.classifier.fit(features, labels)
        self._is_fitted = True

    def predict_proba(self, records: Sequence[ReviewRecord]) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Transformer model is not fitted")
        features = self._build_features(records, fit=False)
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

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "AspectTransformerModel":
        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError("Loaded transformer model has unexpected type")
        model._is_fitted = True
        return model
