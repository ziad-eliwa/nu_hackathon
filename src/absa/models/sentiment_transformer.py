"""Aspect-conditioned sentiment model.

This module keeps the legacy public API name used across the project but now
uses a sparse, class-balanced linear classifier that is more data-efficient for
small and noisy datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from absa.config.taxonomy import SENTIMENT_LABELS, SENTIMENT_TO_ID, is_valid_sentiment
from absa.data.schemas import parse_aspect_sentiments_raw
from absa.preprocess.normalize import normalize_text


def build_aspect_conditioned_text(review_text: str, aspect: str) -> str:
    """Build one model input that conditions sentiment on one aspect."""
    review_text = str(review_text).strip()
    aspect = str(aspect).strip().lower()
    return f"{review_text} [ASPECT] {aspect}"


def normalize_and_correct_text(text: str) -> str:
    """Compatibility wrapper; correction remains optional via normalize_text."""
    return normalize_text(text, apply_spell_correction=True)


def build_training_examples_from_dataframe(
    df: pd.DataFrame,
    text_col: str = "review_text",
    aspect_sentiments_col: str = "aspect_sentiments",
    correct_spelling: bool = False,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Explode review rows into aspect-conditioned sentiment training samples."""
    conditioned_texts: list[str] = []
    labels: list[str] = []
    review_ids: list[str] = []
    aspects: list[str] = []

    for row in df.to_dict(orient="records"):
        review_text = str(row.get(text_col, "")).strip()
        if correct_spelling:
            review_text = normalize_and_correct_text(review_text)
        review_id = str(row.get("review_id", "")).strip()
        sentiment_map = parse_aspect_sentiments_raw(row.get(aspect_sentiments_col))

        for aspect, sentiment in sentiment_map.items():
            if not is_valid_sentiment(sentiment):
                continue

            conditioned_texts.append(build_aspect_conditioned_text(review_text, aspect))
            labels.append(sentiment)
            review_ids.append(review_id)
            aspects.append(aspect)

    return conditioned_texts, labels, review_ids, aspects


@dataclass(slots=True)
class AspectConditionedSentimentModel:
    """Sparse linear sentiment model with aspect-conditioned inputs."""

    word_max_features: int = 40000
    char_max_features: int = 30000
    min_df: int = 2
    random_state: int = 42
    c: float = 2.0
    correct_spelling: bool = False
    class_weight: str = "balanced"
    neutral_weight: float = 1.0

    _word_vectorizer: TfidfVectorizer | None = field(init=False, default=None)
    _char_vectorizer: TfidfVectorizer | None = field(init=False, default=None)
    _classifier: LogisticRegression | None = field(init=False, default=None)
    _single_label: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=self.min_df,
            max_features=self.word_max_features,
            lowercase=False,
        )
        self._char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=self.min_df,
            max_features=self.char_max_features,
            lowercase=False,
        )

        class_weight_value = self.class_weight
        if self.class_weight != "balanced":
            class_weight_value = {"negative": 1.0, "neutral": self.neutral_weight, "positive": 1.0}
        elif self.neutral_weight != 1.0:
            class_weight_value = {0: 1.0, 1: self.neutral_weight, 2: 1.0}

        self._classifier = LogisticRegression(
            C=self.c,
            solver="saga",
            max_iter=1500,
            class_weight=class_weight_value if class_weight_value != "balanced" else "balanced",
            random_state=self.random_state,
        )

    def _transform(self, texts: Sequence[str], fit: bool) -> csr_matrix:
        if self._word_vectorizer is None or self._char_vectorizer is None:
            raise RuntimeError("Vectorizers are not initialized")

        normalized = list(texts)
        if self.correct_spelling:
            normalized = [normalize_and_correct_text(text) for text in normalized]

        if fit:
            word = self._word_vectorizer.fit_transform(normalized)
            char = self._char_vectorizer.fit_transform(normalized)
        else:
            word = self._word_vectorizer.transform(normalized)
            char = self._char_vectorizer.transform(normalized)
        return hstack([word, char]).tocsr()

    def fit(self, conditioned_texts: Sequence[str], labels: Sequence[str]) -> None:
        """Train the model on aspect-conditioned texts."""
        if not conditioned_texts:
            raise ValueError("Cannot train sentiment model on empty training data")

        normalized_labels = [str(label).strip().lower() for label in labels]
        unique_labels = sorted(set(normalized_labels))

        if len(unique_labels) == 1:
            self._single_label = unique_labels[0]
            self._classifier = None
            return

        self._single_label = None
        X = self._transform(conditioned_texts, fit=True)
        y = np.array([SENTIMENT_TO_ID[label] for label in normalized_labels], dtype=np.int32)

        if self._classifier is None:
            raise RuntimeError("Classifier is not initialized")
        self._classifier.fit(X, y)

    def fit_from_dataframe(self, df: pd.DataFrame) -> None:
        """Convenience wrapper to train directly from labeled dataframe."""
        texts, labels, _, _ = build_training_examples_from_dataframe(
            df, correct_spelling=self.correct_spelling
        )
        self.fit(texts, labels)

    def predict(self, review_text: str, aspect: str) -> str:
        """Predict sentiment for one (review, aspect) pair."""
        conditioned = build_aspect_conditioned_text(review_text, aspect)
        return self.predict_conditioned_texts([conditioned])[0]

    def predict_many(self, review_texts: Sequence[str], aspects: Sequence[str]) -> list[str]:
        """Batch sentiment predictions for aligned review/aspect arrays."""
        if len(review_texts) != len(aspects):
            raise ValueError("review_texts and aspects must have same length")

        conditioned = [
            build_aspect_conditioned_text(text, aspect)
            for text, aspect in zip(review_texts, aspects)
        ]
        return self.predict_conditioned_texts(conditioned)

    def predict_conditioned_texts(self, conditioned_texts: Sequence[str]) -> list[str]:
        """Predict sentiments from already-conditioned input strings."""
        if self._single_label is not None:
            return [self._single_label for _ in conditioned_texts]

        if self._classifier is None:
            raise RuntimeError("Model is not fitted")

        X = self._transform(conditioned_texts, fit=False)
        pred_ids = self._classifier.predict(X)
        id_to_sentiment = {v: k for k, v in SENTIMENT_TO_ID.items()}
        return [id_to_sentiment[int(pred)] for pred in pred_ids]

    def predict_proba(self, review_text: str, aspect: str) -> dict[str, float]:
        """Return sentiment probabilities as {label: probability}."""
        conditioned = build_aspect_conditioned_text(review_text, aspect)

        if self._single_label is not None:
            return {
                label: (1.0 if label == self._single_label else 0.0)
                for label in SENTIMENT_LABELS
            }

        if self._classifier is None:
            raise RuntimeError("Model is not fitted")

        X = self._transform([conditioned], fit=False)
        probs = self._classifier.predict_proba(X)[0]

        result = {label: 0.0 for label in SENTIMENT_LABELS}
        for idx, label in enumerate(SENTIMENT_LABELS):
            result[label] = float(probs[idx])
        return result

    def save(self, output_path: str | Path) -> None:
        """Persist model artifact to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "model_type": "sparse_logreg",
            "word_max_features": self.word_max_features,
            "char_max_features": self.char_max_features,
            "min_df": self.min_df,
            "random_state": self.random_state,
            "c": self.c,
            "correct_spelling": self.correct_spelling,
            "class_weight": self.class_weight,
            "neutral_weight": self.neutral_weight,
            "single_label": self._single_label,
            "word_vectorizer": self._word_vectorizer,
            "char_vectorizer": self._char_vectorizer,
            "classifier": self._classifier,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, model_path: str | Path) -> "AspectConditionedSentimentModel":
        """Load model artifact from disk."""
        payload = joblib.load(Path(model_path))

        model = cls(
            word_max_features=int(payload.get("word_max_features", 40000)),
            char_max_features=int(payload.get("char_max_features", 30000)),
            min_df=int(payload.get("min_df", 2)),
            random_state=int(payload.get("random_state", 42)),
            c=float(payload.get("c", 2.0)),
            correct_spelling=bool(payload.get("correct_spelling", False)),
            class_weight=str(payload.get("class_weight", "balanced")),
            neutral_weight=float(payload.get("neutral_weight", 1.0)),
        )

        model._single_label = payload.get("single_label")
        model._word_vectorizer = payload.get("word_vectorizer", model._word_vectorizer)
        model._char_vectorizer = payload.get("char_vectorizer", model._char_vectorizer)
        model._classifier = payload.get("classifier")

        # Backward compatibility with older artifacts: keep a safe hard failure.
        if payload.get("model_state_dict") is not None and model._classifier is None:
            raise ValueError(
                "Legacy neural sentiment artifact detected. Retrain sentiment model "
                "with the current code to use this improved pipeline."
            )

        return model
