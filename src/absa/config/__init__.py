"""Configuration modules for ABSA."""

"""Shared configuration modules.

These files are intentionally small and easy to merge with Path A work.
"""

from .settings import InferenceSettings
from .taxonomy import (
    ASPECT_TAXONOMY,
    ASPECT_TO_INDEX,
    ID_TO_SENTIMENT,
    NONE_ASPECT,
    SENTIMENT_LABELS,
    SENTIMENT_TO_ID,
    is_valid_aspect,
    is_valid_sentiment,
    sorted_unique_aspects,
)

__all__ = [
    "ASPECT_TAXONOMY",
    "ASPECT_TO_INDEX",
    "ID_TO_SENTIMENT",
    "InferenceSettings",
    "NONE_ASPECT",
    "SENTIMENT_LABELS",
    "SENTIMENT_TO_ID",
    "is_valid_aspect",
    "is_valid_sentiment",
    "sorted_unique_aspects",
]
