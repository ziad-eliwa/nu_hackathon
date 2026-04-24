from __future__ import annotations

from typing import Iterable

ASPECTS: tuple[str, ...] = (
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

