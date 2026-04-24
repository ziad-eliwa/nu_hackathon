from __future__ import annotations

import re
import unicodedata

ARABIC_NORMALIZATION_MAP = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ى": "ي",
        "ؤ": "و",
        "ئ": "ي",
    }
)
TATWEEL = "ـ"
WHITESPACE_RE = re.compile(r"\s+")
REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "")
    value = value.replace(TATWEEL, "")
    value = value.translate(ARABIC_NORMALIZATION_MAP)
    value = REPEATED_CHAR_RE.sub(r"\1\1", value)
    value = WHITESPACE_RE.sub(" ", value).strip()
    return value

