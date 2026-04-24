"""Constraint engine for final ABSA predictions.

This module centralizes all hard output rules so they are applied exactly once,
in one place, before any JSON submission is generated.
"""

from __future__ import annotations

from absa.config.taxonomy import NONE_ASPECT, SENTIMENT_LABELS, sorted_unique_aspects
from absa.data.schemas import PredictionRecord, ensure_valid_prediction


def _normalize_sentiment(label: str | None) -> str:
    """Normalize invalid or missing labels to neutral for safety."""
    if not label:
        return "neutral"
    value = str(label).strip().lower()
    if value in SENTIMENT_LABELS:
        return value
    return "neutral"


def finalize_prediction(
    review_id: str,
    aspects: list[str],
    aspect_sentiments: dict[str, str],
) -> PredictionRecord:
    """Apply deterministic cleanup and all hard output constraints.

    Cleanup steps:
    1. Keep only canonical aspects.
    2. Deduplicate and sort by taxonomy order.
    3. Ensure one sentiment per aspect.
    4. Enforce the `none` exclusivity rule.
    """
    # Step 1 and 2: canonicalize aspect list.
    cleaned_aspects = sorted_unique_aspects([str(aspect).strip().lower() for aspect in aspects])

    # If everything was filtered out, fall back to none by design.
    if not cleaned_aspects:
        cleaned_aspects = [NONE_ASPECT]

    # Step 3: keep only sentiments for selected aspects, fill missing with neutral.
    cleaned_sentiments: dict[str, str] = {}
    for aspect in cleaned_aspects:
        raw_value = aspect_sentiments.get(aspect)
        cleaned_sentiments[aspect] = _normalize_sentiment(raw_value)

    # Step 4: hard policy for none.
    if NONE_ASPECT in cleaned_aspects:
        cleaned_aspects = [NONE_ASPECT]
        cleaned_sentiments = {NONE_ASPECT: "neutral"}

    prediction = PredictionRecord(
        review_id=str(review_id),
        aspects=cleaned_aspects,
        aspect_sentiments=cleaned_sentiments,
    )
    return ensure_valid_prediction(prediction)
