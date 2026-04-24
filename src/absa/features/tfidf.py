from __future__ import annotations

import pickle
from pathlib import Path

from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfFeatureExtractor:
    def __init__(
        self,
        max_word_features: int = 40_000,
        max_char_features: int = 20_000,
    ) -> None:
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=max_word_features,
            lowercase=False,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=max_char_features,
            lowercase=False,
        )
        self.metadata_vectorizer = DictVectorizer(sparse=True)

    def fit_transform(
        self, texts: list[str], metadata_dicts: list[dict[str, object]]
    ) -> csr_matrix:
        word = self.word_vectorizer.fit_transform(texts)
        char = self.char_vectorizer.fit_transform(texts)
        meta = self.metadata_vectorizer.fit_transform(metadata_dicts)
        return hstack([word, char, meta]).tocsr()

    def transform(self, texts: list[str], metadata_dicts: list[dict[str, object]]) -> csr_matrix:
        word = self.word_vectorizer.transform(texts)
        char = self.char_vectorizer.transform(texts)
        meta = self.metadata_vectorizer.transform(metadata_dicts)
        return hstack([word, char, meta]).tocsr()

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "TfidfFeatureExtractor":
        with Path(path).open("rb") as handle:
            loaded = pickle.load(handle)
        if not isinstance(loaded, cls):
            raise TypeError("Loaded feature extractor has unexpected type")
        return loaded
