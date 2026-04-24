"""End-to-end ABSA inference orchestration.

This module is the Path B integration point with Path A outputs:
- Path A provides aspect probabilities (plus calibrated thresholds).
- Path B provides sentiment model + postprocess constraints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

import pandas as pd

from absa.config.settings import InferenceSettings
from absa.config.taxonomy import ASPECT_TAXONOMY, NONE_ASPECT
from absa.data.schemas import PredictionRecord, ReviewInput, review_from_mapping
from absa.inference.postprocess import finalize_prediction
from absa.models.sentiment_transformer import AspectConditionedSentimentModel


class AspectProbabilityProvider(Protocol):
    """Path A contract: provider that returns aspect probabilities per review."""

    def predict_aspect_probs(
        self,
        reviews: Sequence[ReviewInput],
    ) -> Sequence[Mapping[str, float]]:
        ...


@dataclass(slots=True)
class JsonAspectProbabilityProvider:
    """File-backed probability provider for quick integration and replay.

    Supported JSON payload shapes:
    1) {"<review_id>": {"service": 0.8, ...}, ...}
    2) [{"review_id": "...", "aspect_probs": {...}}, ...]
    """

    payload_path: str | Path

    def __post_init__(self) -> None:
        payload = json.loads(Path(self.payload_path).read_text(encoding="utf-8"))
        self._by_review_id = self._normalize_payload(payload)

    def _normalize_payload(self, payload: object) -> dict[str, dict[str, float]]:
        by_review_id: dict[str, dict[str, float]] = {}

        if isinstance(payload, dict):
            # Shape 1: keyed dictionary.
            for review_id, aspect_probs in payload.items():
                if isinstance(aspect_probs, Mapping):
                    by_review_id[str(review_id)] = {
                        str(aspect): float(prob)
                        for aspect, prob in aspect_probs.items()
                        if aspect in ASPECT_TAXONOMY
                    }
            return by_review_id

        if isinstance(payload, list):
            # Shape 2: row list.
            for row in payload:
                if not isinstance(row, Mapping):
                    continue
                review_id = str(row.get("review_id", ""))
                aspect_probs = row.get("aspect_probs", row.get("probabilities", {}))
                if review_id and isinstance(aspect_probs, Mapping):
                    by_review_id[review_id] = {
                        str(aspect): float(prob)
                        for aspect, prob in aspect_probs.items()
                        if aspect in ASPECT_TAXONOMY
                    }
            return by_review_id

        raise ValueError("Unsupported aspect probability JSON format")

    def predict_aspect_probs(
        self,
        reviews: Sequence[ReviewInput],
    ) -> Sequence[Mapping[str, float]]:
        # Return empty dict when a review has no Path A probabilities yet.
        return [self._by_review_id.get(review.review_id, {}) for review in reviews]


@dataclass(slots=True)
class CallableAspectProbabilityProvider:
    """Adapter for passing an in-memory Python callable as Path A provider."""

    fn: Callable[[Sequence[ReviewInput]], Sequence[Mapping[str, float]]]

    def predict_aspect_probs(
        self,
        reviews: Sequence[ReviewInput],
    ) -> Sequence[Mapping[str, float]]:
        return self.fn(reviews)


class ABSAPredictor:
    """Path B predictor that combines Path A aspect detection with sentiment."""

    def __init__(
        self,
        sentiment_model: AspectConditionedSentimentModel,
        aspect_provider: AspectProbabilityProvider,
        settings: InferenceSettings | None = None,
    ) -> None:
        self.sentiment_model = sentiment_model
        self.aspect_provider = aspect_provider
        self.settings = settings or InferenceSettings()

    def _is_short_text(self, text: str) -> bool:
        word_count = len(text.split())
        return word_count <= 3

    def _is_mixed_sentiment(self, text: str) -> bool:
        positive_indicators = ['جميل', 'ممتاز', 'عظيم', 'م沂好', 'حب', 'رائع', 'GOOD', 'great', 'love', 'excellent', '👍', '❤', '⭐']
        negative_indicators = ['سيء', 'فظيع', 'مزعج', ' плохо', 'BAD', 'terrible', 'hate', 'worst', '👎', '😡']
        
        text_lower = text.lower()
        pos_count = sum(1 for w in positive_indicators if w in text_lower)
        neg_count = sum(1 for w in negative_indicators if w in text_lower)
        
        return pos_count > 0 and neg_count > 0

    def _select_aspects(self, aspect_probs: Mapping[str, float], review_text: str = "") -> list[str]:
        """Convert probability map into final aspect list with calibrated rules."""
        normalized_probs = {
            aspect: float(aspect_probs.get(aspect, 0.0)) for aspect in ASPECT_TAXONOMY
        }

        is_short = self._is_short_text(review_text) if review_text else False
        is_mixed = self._is_mixed_sentiment(review_text) if review_text else False

        if is_short and review_text:
            word_count = len(review_text.split())
            if word_count <= 2:
                none_prob = normalized_probs.get(NONE_ASPECT, 0)
                general_prob = normalized_probs.get("general", 0)
                if none_prob > 0.3 or general_prob > 0.4:
                    return [NONE_ASPECT] if none_prob > general_prob else ["general"]

        selected_non_none = [
            aspect
            for aspect in ASPECT_TAXONOMY
            if aspect != NONE_ASPECT
            and normalized_probs[aspect] >= self.settings.threshold_for(aspect)
        ]

        if selected_non_none:
            return selected_non_none

        if normalized_probs[NONE_ASPECT] >= self.settings.threshold_for(NONE_ASPECT):
            return [NONE_ASPECT]

        if self.settings.fallback_aspect in ASPECT_TAXONOMY:
            return [self.settings.fallback_aspect]

        return [NONE_ASPECT]

    def predict_reviews(self, reviews: Sequence[ReviewInput]) -> list[PredictionRecord]:
        """Run full Path B inference for a batch of reviews."""
        aspect_prob_list = list(self.aspect_provider.predict_aspect_probs(reviews))
        if len(aspect_prob_list) != len(reviews):
            raise ValueError(
                "Path A provider returned different number of rows than input reviews"
            )

        outputs: list[PredictionRecord] = []

        for review, aspect_probs in zip(reviews, aspect_prob_list):
            chosen_aspects = self._select_aspects(aspect_probs, review.review_text)

            sentiment_map: dict[str, str] = {}
            for aspect in chosen_aspects:
                if aspect == NONE_ASPECT:
                    sentiment_map[aspect] = "neutral"
                    continue

                sentiment_map[aspect] = self.sentiment_model.predict(
                    review_text=review.review_text,
                    aspect=aspect,
                )

            outputs.append(
                finalize_prediction(
                    review_id=review.review_id,
                    aspects=chosen_aspects,
                    aspect_sentiments=sentiment_map,
                )
            )

        return outputs

    @classmethod
    def from_artifacts(
        cls,
        sentiment_model_path: str | Path,
        aspect_provider: AspectProbabilityProvider,
        threshold_file: str | Path | None = None,
        default_threshold: float = 0.5,
        fallback_aspect: str = "general",
    ) -> "ABSAPredictor":
        """Convenience constructor for file-based production-style loading."""
        sentiment_model = AspectConditionedSentimentModel.load(sentiment_model_path)

        if threshold_file is not None:
            settings = InferenceSettings.from_threshold_file(
                threshold_path=threshold_file,
                default_threshold=default_threshold,
                fallback_aspect=fallback_aspect,
            )
        else:
            settings = InferenceSettings(
                default_threshold=default_threshold,
                fallback_aspect=fallback_aspect,
            )

        return cls(
            sentiment_model=sentiment_model,
            aspect_provider=aspect_provider,
            settings=settings,
        )


def reviews_from_dataframe(df: pd.DataFrame) -> list[ReviewInput]:
    """Convert a dataframe to typed review records used by predictor APIs."""
    return [review_from_mapping(row) for row in df.to_dict(orient="records")]
