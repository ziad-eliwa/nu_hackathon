from __future__ import annotations

from pathlib import Path

from absa.data.io import load_labeled_reviews, load_unlabeled_reviews
from absa.preprocess.normalize import normalize_text


def test_load_labeled_reviews_reads_known_file():
    path = Path("data/DeepX_validation.csv")
    records = load_labeled_reviews(path)
    assert len(records) > 0
    assert records[0].review_id
    assert records[0].aspects


def test_load_unlabeled_reviews_reads_known_file():
    path = Path("data/DeepX_train.csv")
    records = load_unlabeled_reviews(path)
    assert len(records) > 0
    assert records[0].review_text


def test_normalize_text_arabic_cleanup():
    text = "آآآهــلاااا   بكم"
    normalized = normalize_text(text)
    assert "ـ" not in normalized
    assert "آ" not in normalized
    assert "  " not in normalized

