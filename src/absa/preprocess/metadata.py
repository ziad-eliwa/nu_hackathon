from __future__ import annotations

from typing import Any

from sklearn.feature_extraction import DictVectorizer

from absa.data.schemas import ReviewRecord


def record_to_metadata(record: ReviewRecord) -> dict[str, Any]:
    return {
        "platform": record.platform,
        "business_category": record.business_category,
        "star_rating": record.star_rating,
    }


def build_metadata_matrix(
    records: list[ReviewRecord],
    vectorizer: DictVectorizer | None = None,
    fit: bool = False,
):
    metadata = [record_to_metadata(record) for record in records]
    if vectorizer is None:
        vectorizer = DictVectorizer(sparse=True)
        fit = True
    if fit:
        matrix = vectorizer.fit_transform(metadata)
    else:
        matrix = vectorizer.transform(metadata)
    return matrix, vectorizer

