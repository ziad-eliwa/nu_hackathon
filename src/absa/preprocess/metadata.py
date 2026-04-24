from __future__ import annotations

import re
from typing import Any

from sklearn.feature_extraction import DictVectorizer

from absa.data.schemas import ReviewRecord


def record_to_metadata(record: ReviewRecord) -> dict[str, Any]:
    text = record.review_text
    word_count = len(text.split())
    char_count = len(text)
    safe_char_count = max(1, char_count)
    
    is_short = word_count <= 3
    is_medium = 4 <= word_count <= 15
    is_long = word_count > 15
    
    has_emoji = bool(re.search(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', text))
    has_arabic = bool(re.search(r'[\u0600-\u06FF]', text))
    has_english = bool(re.search(r'[a-zA-Z]', text))
    is_mixed = has_arabic and has_english
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    digit_chars = len(re.findall(r'[0-9٠-٩]', text))
    emoji_count = len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', text))
    elongated_count = len(re.findall(r'(.)\1{2,}', text))
    
    exclamation_count = text.count('!')
    question_count = text.count('?')
    arabic_ratio = arabic_chars / safe_char_count
    latin_ratio = latin_chars / safe_char_count
    digit_ratio = digit_chars / safe_char_count
    
    return {
        "platform": record.platform,
        "business_category": record.business_category,
        "star_rating": record.star_rating,
        "word_count": word_count,
        "char_count": char_count,
        "is_short": is_short,
        "is_medium": is_medium,
        "is_long": is_long,
        "has_emoji": has_emoji,
        "emoji_count": emoji_count,
        "is_mixed_language": is_mixed,
        "arabic_ratio": arabic_ratio,
        "latin_ratio": latin_ratio,
        "digit_ratio": digit_ratio,
        "elongated_count": elongated_count,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
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

