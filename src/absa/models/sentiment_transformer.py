"""Aspect-conditioned sentiment model.

Why this file is named `sentiment_transformer`:
- The architecture target for Path B is a transformer sentiment model.
- This implementation provides a drop-in baseline API (TF-IDF + Logistic)
  so the pipeline is operational now and can later be upgraded to AraBERT
  without changing the rest of Path B.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from absa.config.taxonomy import SENTIMENT_LABELS, is_valid_sentiment
from absa.data.schemas import parse_aspect_sentiments_raw


def build_aspect_conditioned_text(review_text: str, aspect: str) -> str:
    """Build one model input that conditions sentiment on an aspect.

    We mimic transformer-style paired input by appending the aspect with a
    special separator token marker.
    """
    review_text = str(review_text).strip()
    aspect = str(aspect).strip().lower()
    return f"{review_text} [ASPECT] {aspect}"


def build_training_examples_from_dataframe(
    df: pd.DataFrame,
    text_col: str = "review_text",
    aspect_sentiments_col: str = "aspect_sentiments",
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Explode review rows into aspect-conditioned sentiment training samples.

    Returns:
        conditioned_texts: model inputs
        labels: sentiment labels
        review_ids: review IDs for traceability
        aspects: aspects aligned to each sample
    """
    conditioned_texts: list[str] = []
    labels: list[str] = []
    review_ids: list[str] = []
    aspects: list[str] = []

    # Iterate row-by-row because each row can contain multiple aspect labels.
    for row in df.to_dict(orient="records"):
        review_text = str(row.get(text_col, "")).strip()
        review_id = str(row.get("review_id", "")).strip()
        sentiment_map = parse_aspect_sentiments_raw(row.get(aspect_sentiments_col))

        # One review contributes one sample per labeled aspect sentiment.
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
    """Trainable sentiment model with a stable interface for inference.

    Notes:
    - The `pipeline` can be replaced by a transformer encoder in future.
    - We keep `predict` and `predict_many` methods stable for Path B integration.
    """

    max_features: int = 60000
    min_df: int = 2
    ngram_range: tuple[int, int] = (3, 5)
    random_state: int = 42
    _single_label: str | None = field(init=False, default=None)
    _pipeline: Pipeline | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._pipeline: Pipeline | None = Pipeline(
            steps=[
                (
                    "vectorizer",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=self.ngram_range,
                        min_df=self.min_df,
                        max_features=self.max_features,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=400,
                        class_weight="balanced",
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def fit(self, conditioned_texts: Sequence[str], labels: Sequence[str]) -> None:
        """Fit model on already prepared aspect-conditioned texts."""
        if not conditioned_texts:
            raise ValueError("Cannot train sentiment model on empty training data")

        normalized_labels = [str(label).strip().lower() for label in labels]
        unique_labels = sorted(set(normalized_labels))

        # Degenerate fallback: if dataset only has one class, avoid sklearn error.
        if len(unique_labels) == 1:
            self._single_label = unique_labels[0]
            self._pipeline = None
            return

        if self._pipeline is None:
            raise RuntimeError("Pipeline is missing and no single-label fallback is set")

        self._single_label = None
        self._pipeline.fit(list(conditioned_texts), normalized_labels)

    def fit_from_dataframe(self, df: pd.DataFrame) -> None:
        """Convenience wrapper to train directly from labeled dataframe."""
        texts, labels, _, _ = build_training_examples_from_dataframe(df)
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

        if self._pipeline is None:
            raise RuntimeError("Model is not fitted")

        predictions = self._pipeline.predict(list(conditioned_texts))
        return [str(prediction) for prediction in predictions]

    def predict_proba(self, review_text: str, aspect: str) -> dict[str, float]:
        """Return sentiment probabilities as {label: probability}."""
        conditioned = build_aspect_conditioned_text(review_text, aspect)

        if self._single_label is not None:
            return {
                label: (1.0 if label == self._single_label else 0.0)
                for label in SENTIMENT_LABELS
            }

        if self._pipeline is None:
            raise RuntimeError("Model is not fitted")

        probabilities = self._pipeline.predict_proba([conditioned])[0]
        labels = [str(label) for label in self._pipeline.classes_]

        result = {label: 0.0 for label in SENTIMENT_LABELS}
        for label, prob in zip(labels, probabilities):
            result[label] = float(prob)
        return result

    def save(self, output_path: str | Path) -> None:
        """Persist model artifact to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "max_features": self.max_features,
            "min_df": self.min_df,
            "ngram_range": self.ngram_range,
            "random_state": self.random_state,
            "single_label": self._single_label,
            "pipeline": self._pipeline,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, model_path: str | Path) -> "AspectConditionedSentimentModel":
        """Load model artifact from disk."""
        payload = joblib.load(Path(model_path))

        model = cls(
            max_features=int(payload["max_features"]),
            min_df=int(payload["min_df"]),
            ngram_range=tuple(payload["ngram_range"]),
            random_state=int(payload["random_state"]),
        )
        model._single_label = payload["single_label"]
        model._pipeline = payload["pipeline"]
        return model
