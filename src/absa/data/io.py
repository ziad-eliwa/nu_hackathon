from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd

from absa.config.taxonomy import ordered_aspects
from absa.data.schemas import LabeledReviewRecord, ReviewRecord, parse_aspect_sentiments, parse_aspects, validate_labeled_record


def _parse_jsonish(value: str) -> Any:
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def _as_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return fallback


def load_unlabeled_reviews(csv_path: str | Path) -> list[ReviewRecord]:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    records: list[ReviewRecord] = []
    for row in df.to_dict(orient="records"):
        records.append(
            ReviewRecord(
                review_id=str(row["review_id"]),
                review_text=str(row["review_text"]),
                star_rating=_as_int(row.get("star_rating", 0)),
                date=str(row.get("date", "")),
                business_name=str(row.get("business_name", "")),
                business_category=str(row.get("business_category", "")),
                platform=str(row.get("platform", "")),
            )
        )
    return records


def load_labeled_reviews(csv_path: str | Path) -> list[LabeledReviewRecord]:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    records: list[LabeledReviewRecord] = []
    for row in df.to_dict(orient="records"):
        aspects_raw = _parse_jsonish(str(row.get("aspects", "[]")))
        sentiments_raw = _parse_jsonish(str(row.get("aspect_sentiments", "{}")))
        aspects = tuple(parse_aspects(aspects_raw))
        sentiments = parse_aspect_sentiments(sentiments_raw)
        record = LabeledReviewRecord(
            review_id=str(row["review_id"]),
            review_text=str(row["review_text"]),
            star_rating=_as_int(row.get("star_rating", 0)),
            date=str(row.get("date", "")),
            business_name=str(row.get("business_name", "")),
            business_category=str(row.get("business_category", "")),
            platform=str(row.get("platform", "")),
            aspects=tuple(ordered_aspects(aspects)),
            aspect_sentiments=sentiments,
        )
        validate_labeled_record(record)
        records.append(record)
    return records


def labeled_records_to_dataframe(records: list[LabeledReviewRecord]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "review_id": record.review_id,
                "review_text": record.review_text,
                "star_rating": record.star_rating,
                "date": record.date,
                "business_name": record.business_name,
                "business_category": record.business_category,
                "platform": record.platform,
                "aspects": list(record.aspects),
                "aspect_sentiments": dict(record.aspect_sentiments),
            }
        )
    return pd.DataFrame(rows)

