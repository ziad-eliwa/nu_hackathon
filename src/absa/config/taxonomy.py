from __future__ import annotations

from typing import Final

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
CONCRETE_ASPECTS: tuple[str, ...] = tuple(a for a in ASPECT_TAXONOMY if a != "none")
SENTIMENT_LABELS: Final[tuple[str, ...]] = ("negative", "neutral", "positive")
ASPECT_ORDER: dict[str, int] = {aspect: idx for idx, aspect in enumerate(ASPECT_TAXONOMY)}

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
    return value in ASPECT_TO_INDEX


def is_valid_sentiment(value: str) -> bool:
    return value in SENTIMENT_TO_ID


def sorted_unique_aspects(aspects: list[str]) -> list[str]:
    unique = {aspect for aspect in aspects if is_valid_aspect(aspect)}
    return [aspect for aspect in ASPECT_TAXONOMY if aspect in unique]


def ordered_aspects(aspects: list[str]) -> list[str]:
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


def assert_none_exclusive(aspects: list[str]) -> None:
    if "none" in aspects and len(aspects) > 1:
        raise ValueError("'none' must not appear with other aspects")


NONE_ASPECT: Final[str] = "none"