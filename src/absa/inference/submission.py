"""Submission packaging utilities.

This module converts typed predictions to strict JSON rows and validates each
record before writing the final file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from absa.data.schemas import PredictionRecord, ensure_valid_prediction, prediction_to_dict


def build_submission_rows(
    predictions: Sequence[PredictionRecord],
    validate: bool = True,
) -> list[dict]:
    """Convert prediction objects into JSON-serializable row dictionaries."""
    rows: list[dict] = []
    for prediction in predictions:
        if validate:
            ensure_valid_prediction(prediction)
        rows.append(prediction_to_dict(prediction))
    return rows


def write_submission_json(
    predictions: Sequence[PredictionRecord],
    output_path: str | Path,
    validate: bool = True,
    indent: int = 2,
) -> Path:
    """Write strict submission JSON to disk and return path."""
    rows = build_submission_rows(predictions, validate=validate)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )
    return path
