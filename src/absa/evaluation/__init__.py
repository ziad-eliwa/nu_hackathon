"""Evaluation and error analysis helpers."""

from .error_analysis import (
    build_comparison_frame,
    standard_slice_reports,
    top_failure_examples,
)
from .metrics import (
    compute_aspect_detection_f1,
    compute_sentiment_macro_f1_given_aspect,
    compute_tuple_f1,
    evaluate_predictions,
)

__all__ = [
    "build_comparison_frame",
    "compute_aspect_detection_f1",
    "compute_sentiment_macro_f1_given_aspect",
    "compute_tuple_f1",
    "evaluate_predictions",
    "standard_slice_reports",
    "top_failure_examples",
]
