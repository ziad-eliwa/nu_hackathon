from __future__ import annotations

from typing import Iterable

ASPECTS: tuple[str, ...] = (
"""Canonical taxonomy and label helpers.

This file is shared infrastructure for both Path A and Path B. Keeping one
source of truth here avoids subtle mismatches in label ordering.
"""

from __future__ import annotations

from typing import Final

# The aspect order is fixed by the task and must never be changed ad hoc.
ASPECT_TAXONOMY: Final[tuple[str, ...]] = (
    "food",
    "service",
    "price",
    "cleanliness",
    "delivery",
    "ambiance",
    "app_experience",
    "general",
    "none",
)
CONCRETE_ASPECTS: tuple[str, ...] = tuple(a for a in ASPECTS if a != "none")
SENTIMENT_LABELS: tuple[str, ...] = ("positive", "negative", "neutral")
ASPECT_ORDER: dict[str, int] = {aspect: idx for idx, aspect in enumerate(ASPECTS)}


def is_valid_aspect(aspect: str) -> bool:
    return aspect in ASPECT_ORDER


def is_valid_sentiment(sentiment: str) -> bool:
    return sentiment in SENTIMENT_LABELS


def ordered_aspects(aspects: Iterable[str]) -> list[str]:
    unique = []
    seen = set()
    for aspect in aspects:
        if aspect in seen:
            continue
        if not is_valid_aspect(aspect):
            raise ValueError(f"Unknown aspect: {aspect}")
        seen.add(aspect)
        unique.append(aspect)
    return sorted(unique, key=ASPECT_ORDER.__getitem__)


def assert_none_exclusive(aspects: Iterable[str]) -> None:
    values = list(aspects)
    if "none" in values and len(values) > 1:
        raise ValueError("'none' must not appear with other aspects")


# Special aspect with exclusivity constraints.
NONE_ASPECT: Final[str] = "none"

# Sentiment labels are the only valid classes for aspect sentiment.
SENTIMENT_LABELS: Final[tuple[str, ...]] = ("negative", "neutral", "positive")

ASPECT_TO_INDEX: Final[dict[str, int]] = {
    aspect: idx for idx, aspect in enumerate(ASPECT_TAXONOMY)
}
SENTIMENT_TO_ID: Final[dict[str, int]] = {
    sentiment: idx for idx, sentiment in enumerate(SENTIMENT_LABELS)
}
ID_TO_SENTIMENT: Final[dict[int, str]] = {
    idx: sentiment for sentiment, idx in SENTIMENT_TO_ID.items()
}


def is_valid_aspect(value: str) -> bool:
    """Return True only for canonical aspect labels."""
    return value in ASPECT_TO_INDEX


def is_valid_sentiment(value: str) -> bool:
    """Return True only for canonical sentiment labels."""
    return value in SENTIMENT_TO_ID


def sorted_unique_aspects(aspects: list[str]) -> list[str]:
    """Deduplicate and sort aspects by canonical taxonomy order.

    We intentionally keep deterministic ordering to simplify debugging,
    reproducibility, and downstream JSON comparisons.
    """
    unique = {aspect for aspect in aspects if is_valid_aspect(aspect)}
    return [aspect for aspect in ASPECT_TAXONOMY if aspect in unique]
