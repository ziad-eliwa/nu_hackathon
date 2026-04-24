from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from absa.config.taxonomy import (
    SENTIMENT_LABELS,
    assert_none_exclusive,
    is_valid_aspect,
    is_valid_sentiment,
    ordered_aspects,
)


@dataclass(frozen=True)
class ReviewRecord:
    review_id: str
    review_text: str
    star_rating: int
    date: str
    business_name: str
    business_category: str
    platform: str


@dataclass(frozen=True)
class LabeledReviewRecord(ReviewRecord):
    aspects: tuple[str, ...]
    aspect_sentiments: dict[str, str]


def parse_aspects(raw: Any) -> list[str]:
    if isinstance(raw, list):
        aspects = [str(item).strip() for item in raw if str(item).strip()]
        return ordered_aspects(aspects)
    raise ValueError(f"Invalid aspects payload type: {type(raw)}")


def parse_aspect_sentiments(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid aspect_sentiments payload type: {type(raw)}")
    normalized: dict[str, str] = {}
    for key, value in raw.items():
        aspect = str(key).strip()
        sentiment = str(value).strip().lower()
        if not is_valid_aspect(aspect):
            raise ValueError(f"Unknown aspect in aspect_sentiments: {aspect}")
        if not is_valid_sentiment(sentiment):
            raise ValueError(f"Unknown sentiment for {aspect}: {sentiment}")
        normalized[aspect] = sentiment
    return normalized


def validate_labeled_record(record: LabeledReviewRecord) -> None:
    if not record.review_id:
        raise ValueError("review_id cannot be empty")
    if not record.review_text:
        raise ValueError(f"review_text cannot be empty for review_id={record.review_id}")
    for aspect in record.aspects:
        if not is_valid_aspect(aspect):
            raise ValueError(f"Unknown aspect {aspect} in review_id={record.review_id}")
    assert_none_exclusive(record.aspects)
    if set(record.aspects) != set(record.aspect_sentiments):
        raise ValueError(
            f"aspect_sentiments keys must equal aspects for review_id={record.review_id}"
        )
    for sentiment in record.aspect_sentiments.values():
        if sentiment not in SENTIMENT_LABELS:
            raise ValueError(
                f"Invalid sentiment in review_id={record.review_id}: {sentiment}"
            )
    if "none" in record.aspects and record.aspect_sentiments.get("none") != "neutral":
        raise ValueError("'none' must have neutral sentiment")

