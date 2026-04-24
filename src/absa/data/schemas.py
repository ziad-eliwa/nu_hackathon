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

"""Schema and validation helpers.

This module gives Path B strict runtime contracts for:
- input review records
- output prediction records
- parsing CSV string payloads for aspects and sentiment maps
"""

from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from absa.config.taxonomy import (
    NONE_ASPECT,
    SENTIMENT_LABELS,
    is_valid_aspect,
    is_valid_sentiment,
    sorted_unique_aspects,
)


def _is_missing(value: Any) -> bool:
    """Treat None/NaN as missing without importing pandas in this module."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _try_parse_collection(text: str) -> Any:
    """Parse JSON-like string using JSON first, then Python literal fallback."""
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            continue
    return None


def parse_aspects_raw(value: Any) -> list[str]:
    """Parse aspects value from CSV/object into normalized list[str].

    Examples accepted:
    - ["service", "delivery"]
    - "[\"service\", \"delivery\"]"
    - "['service', 'delivery']"
    """
    if _is_missing(value):
        return []

    if isinstance(value, list):
        aspects = [str(item).strip().lower() for item in value if str(item).strip()]
        return sorted_unique_aspects(aspects)

    text = str(value).strip()
    if not text:
        return []

    parsed = _try_parse_collection(text)
    if isinstance(parsed, list):
        aspects = [str(item).strip().lower() for item in parsed if str(item).strip()]
        return sorted_unique_aspects(aspects)

    # Last-resort fallback for malformed list strings.
    text = text.strip("[]")
    if not text:
        return []

    raw_parts = [part.strip().strip('"\'') for part in text.split(",")]
    aspects = [part.lower() for part in raw_parts if part]
    return sorted_unique_aspects(aspects)


def parse_aspect_sentiments_raw(value: Any) -> dict[str, str]:
    """Parse aspect_sentiments payload into {aspect: sentiment}.

    Input commonly comes in CSV-escaped JSON strings, so this function is
    intentionally tolerant but only returns valid canonical labels.
    """
    if _is_missing(value):
        return {}

    if isinstance(value, dict):
        raw_map = value
    else:
        text = str(value).strip().replace('""', '"')
        parsed = _try_parse_collection(text)
        if isinstance(parsed, str):
            parsed = _try_parse_collection(parsed.strip())
        raw_map = parsed if isinstance(parsed, dict) else {}

    result: dict[str, str] = {}
    for raw_aspect, raw_sentiment in raw_map.items():
        aspect = str(raw_aspect).strip().lower()
        sentiment = str(raw_sentiment).strip().lower()
        if is_valid_aspect(aspect) and is_valid_sentiment(sentiment):
            result[aspect] = sentiment
    return result


@dataclass(slots=True)
class ReviewInput:
    """Minimal review object consumed by inference APIs."""

    review_id: str
    review_text: str
    platform: str | None = None
    business_category: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PredictionRecord:
    """Canonical prediction object for one review."""

    review_id: str
    aspects: list[str]
    aspect_sentiments: dict[str, str]


def review_from_mapping(row: Mapping[str, Any]) -> ReviewInput:
    """Build `ReviewInput` from a dictionary-like row."""
    review_id = str(row.get("review_id", "")).strip()
    review_text = str(row.get("review_text", "")).strip()
    return ReviewInput(
        review_id=review_id,
        review_text=review_text,
        platform=(str(row.get("platform")).strip() if row.get("platform") else None),
        business_category=(
            str(row.get("business_category")).strip()
            if row.get("business_category")
            else None
        ),
    )


def _prediction_errors(prediction: PredictionRecord) -> list[str]:
    """Collect all schema violations instead of failing at first error."""
    errors: list[str] = []

    if not prediction.review_id:
        errors.append("review_id must be a non-empty string")

    if len(prediction.aspects) != len(set(prediction.aspects)):
        errors.append("aspects must not contain duplicates")

    for aspect in prediction.aspects:
        if not is_valid_aspect(aspect):
            errors.append(f"invalid aspect '{aspect}'")

    for aspect, sentiment in prediction.aspect_sentiments.items():
        if aspect not in prediction.aspects:
            errors.append(
                f"aspect_sentiments contains key '{aspect}' that is not in aspects"
            )
        if not is_valid_sentiment(sentiment):
            errors.append(f"invalid sentiment '{sentiment}' for aspect '{aspect}'")

    for aspect in prediction.aspects:
        if aspect not in prediction.aspect_sentiments:
            errors.append(
                f"missing sentiment for aspect '{aspect}' in aspect_sentiments"
            )

    # Hard rule from task spec: none is exclusive and always neutral.
    if NONE_ASPECT in prediction.aspects:
        if prediction.aspects != [NONE_ASPECT]:
            errors.append("'none' cannot appear with any other aspect")
        if prediction.aspect_sentiments.get(NONE_ASPECT) != "neutral":
            errors.append("'none' sentiment must always be 'neutral'")

    return errors


def ensure_valid_prediction(prediction: PredictionRecord) -> PredictionRecord:
    """Validate and return the same object when valid, otherwise raise error."""
    errors = _prediction_errors(prediction)
    if errors:
        joined = " | ".join(errors)
        raise ValueError(f"Prediction schema validation failed: {joined}")
    return prediction


def prediction_to_dict(prediction: PredictionRecord) -> dict[str, Any]:
    """Serialize prediction object into submission-friendly dictionary."""
    return {
        "review_id": prediction.review_id,
        "aspects": prediction.aspects,
        "aspect_sentiments": prediction.aspect_sentiments,
    }
