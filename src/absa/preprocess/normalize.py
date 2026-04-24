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

_spell_checker = None


def _get_spell_checker():
    global _spell_checker
    if _spell_checker is None:
        try:
            from spellchecker import SpellChecker
            _spell_checker = SpellChecker(language='ar')
        except Exception:
            _spell_checker = False
    return _spell_checker


def correct_spelling(text: str) -> str:
    spell = _get_spell_checker()
    if not spell:
        return text
    
    words = text.split()
    corrected_words = []
    
    for word in words:
        misspelled = spell.unknown([word])
        if misspelled:
            correction = spell.correction(word)
            if correction:
                corrected_words.append(correction)
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)
    
    return " ".join(corrected_words)


def normalize_text(text: str, apply_spell_correction: bool = False) -> str:
    value = unicodedata.normalize("NFKC", text or "")
    value = value.replace(TATWEEL, "")
    value = value.translate(ARABIC_NORMALIZATION_MAP)
    value = REPEATED_CHAR_RE.sub(r"\1\1", value)
    value = WHITESPACE_RE.sub(" ", value).strip()
    
    if apply_spell_correction:
        value = correct_spelling(value)
    
    return value

