from __future__ import annotations

import pytest

from absa.config.taxonomy import ASPECTS, ordered_aspects
from absa.data.schemas import LabeledReviewRecord, validate_labeled_record


def test_ordered_aspects_is_taxonomy_sorted_and_unique():
    values = ordered_aspects(["service", "food", "service", "none"])
    assert values == ["food", "service", "none"]


def test_validate_labeled_record_none_must_be_exclusive():
    record = LabeledReviewRecord(
        review_id="1",
        review_text="test",
        star_rating=1,
        date="today",
        business_name="x",
        business_category="y",
        platform="google_maps",
        aspects=("none", "food"),
        aspect_sentiments={"none": "neutral", "food": "negative"},
    )
    with pytest.raises(ValueError):
        validate_labeled_record(record)


def test_validate_labeled_record_sentiment_keys_match_aspects():
    record = LabeledReviewRecord(
        review_id="2",
        review_text="test",
        star_rating=2,
        date="today",
        business_name="x",
        business_category="y",
        platform="google_maps",
        aspects=("food",),
        aspect_sentiments={"service": "negative"},
    )
    with pytest.raises(ValueError):
        validate_labeled_record(record)

