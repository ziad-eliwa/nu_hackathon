"""Aspect-conditioned sentiment model using neural network with PyTorch.

Two hidden layer MLP for sentiment classification with spell correction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, TensorDataset

from absa.config.taxonomy import SENTIMENT_LABELS, SENTIMENT_TO_ID, is_valid_sentiment
from absa.data.schemas import parse_aspect_sentiments_raw
from absa.preprocess.normalize import normalize_text


def build_aspect_conditioned_text(review_text: str, aspect: str) -> str:
    """Build one model input that conditions sentiment on an aspect."""
    review_text = str(review_text).strip()
    aspect = str(aspect).strip().lower()
    return f"{review_text} [ASPECT] {aspect}"


def normalize_and_correct_text(text: str) -> str:
    """Apply normalization and spell correction to text."""
    return normalize_text(text, apply_spell_correction=True)


def build_training_examples_from_dataframe(
    df: pd.DataFrame,
    text_col: str = "review_text",
    aspect_sentiments_col: str = "aspect_sentiments",
    correct_spelling: bool = False,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Explode review rows into aspect-conditioned sentiment training samples."""
    conditioned_texts: list[str] = []
    labels: list[str] = []
    review_ids: list[str] = []
    aspects: list[str] = []

    for row in df.to_dict(orient="records"):
        review_text = str(row.get(text_col, "")).strip()
        if correct_spelling:
            review_text = normalize_and_correct_text(review_text)
        review_id = str(row.get("review_id", "")).strip()
        sentiment_map = parse_aspect_sentiments_raw(row.get(aspect_sentiments_col))

        for aspect, sentiment in sentiment_map.items():
            if not is_valid_sentiment(sentiment):
                continue

            conditioned_texts.append(build_aspect_conditioned_text(review_text, aspect))
            labels.append(sentiment)
            review_ids.append(review_id)
            aspects.append(aspect)

    return conditioned_texts, labels, review_ids, aspects


class SentimentMLP(nn.Module):
    """Two-layer MLP for sentiment classification."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass(slots=True)
class AspectConditionedSentimentModel:
    """Neural network sentiment model with PyTorch."""

    max_features: int = 60000
    min_df: int = 2
    ngram_range: tuple[int, int] = (3, 5)
    random_state: int = 42
    hidden_dim: int = 256
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 0.001
    dropout: float = 0.3
    correct_spelling: bool = False
    
    _vectorizer: TfidfVectorizer | None = field(init=False, default=None)
    _model: SentimentMLP | None = field(init=False, default=None)
    _device: str = field(init=False, default="cpu")
    _single_label: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_features=self.max_features,
        )

    def _build_model(self, input_dim: int) -> SentimentMLP:
        return SentimentMLP(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_classes=len(SENTIMENT_LABELS),
            dropout=self.dropout,
        ).to(self._device)

    def fit(self, conditioned_texts: Sequence[str], labels: Sequence[str]) -> None:
        """Train the neural network on aspect-conditioned texts."""
        if not conditioned_texts:
            raise ValueError("Cannot train sentiment model on empty training data")

        normalized_labels = [str(label).strip().lower() for label in labels]
        unique_labels = sorted(set(normalized_labels))

        if len(unique_labels) == 1:
            self._single_label = unique_labels[0]
            self._model = None
            return

        self._single_label = None
        
        texts = list(conditioned_texts)
        if self.correct_spelling:
            texts = [normalize_and_correct_text(t) for t in texts]
        
        X = self._vectorizer.fit_transform(texts).toarray()
        y = np.array([SENTIMENT_TO_ID[label] for label in normalized_labels])
        
        input_dim = X.shape[1]
        self._model = self._build_model(input_dim)
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        class_counts = np.bincount(y)
        class_weights = 1.0 / (class_counts + 1)
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        class_weights = torch.FloatTensor(class_weights).to(self._device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        
        self._model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self._device)
                batch_y = batch_y.to(self._device)
                
                optimizer.zero_grad()
                outputs = self._model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        self._model.eval()

    def fit_from_dataframe(self, df: pd.DataFrame) -> None:
        """Convenience wrapper to train directly from labeled dataframe."""
        texts, labels, _, _ = build_training_examples_from_dataframe(
            df, correct_spelling=self.correct_spelling
        )
        self.fit(texts, labels)

    def predict(self, review_text: str, aspect: str) -> str:
        """Predict sentiment for one (review, aspect) pair."""
        if self.correct_spelling:
            review_text = normalize_and_correct_text(review_text)
        conditioned = build_aspect_conditioned_text(review_text, aspect)
        return self.predict_conditioned_texts([conditioned])[0]

    def predict_many(self, review_texts: Sequence[str], aspects: Sequence[str]) -> list[str]:
        """Batch sentiment predictions for aligned review/aspect arrays."""
        if len(review_texts) != len(aspects):
            raise ValueError("review_texts and aspects must have same length")

        texts = list(review_texts)
        if self.correct_spelling:
            texts = [normalize_and_correct_text(t) for t in texts]

        conditioned = [
            build_aspect_conditioned_text(text, aspect)
            for text, aspect in zip(texts, aspects)
        ]
        return self.predict_conditioned_texts(conditioned)

    def predict_conditioned_texts(self, conditioned_texts: Sequence[str]) -> list[str]:
        """Predict sentiments from already-conditioned input strings."""
        if self._single_label is not None:
            return [self._single_label for _ in conditioned_texts]

        if self._model is None or self._vectorizer is None:
            raise RuntimeError("Model is not fitted")

        texts = list(conditioned_texts)
        if self.correct_spelling:
            texts = [normalize_and_correct_text(t) for t in texts]

        X = self._vectorizer.transform(texts).toarray()
        X_tensor = torch.FloatTensor(X).to(self._device)
        
        self._model.eval()
        with torch.no_grad():
            outputs = self._model(X_tensor)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
        id_to_sentiment = {v: k for k, v in SENTIMENT_TO_ID.items()}
        return [id_to_sentiment[int(pred)] for pred in predictions]

    def predict_proba(self, review_text: str, aspect: str) -> dict[str, float]:
        """Return sentiment probabilities as {label: probability}."""
        if self.correct_spelling:
            review_text = normalize_and_correct_text(review_text)
        conditioned = build_aspect_conditioned_text(review_text, aspect)

        if self._single_label is not None:
            return {
                label: (1.0 if label == self._single_label else 0.0)
                for label in SENTIMENT_LABELS
            }

        if self._model is None or self._vectorizer is None:
            raise RuntimeError("Model is not fitted")

        texts = [conditioned]
        if self.correct_spelling:
            texts = [normalize_and_correct_text(t) for t in texts]

        X = self._vectorizer.transform(texts).toarray()
        X_tensor = torch.FloatTensor(X).to(self._device)
        
        self._model.eval()
        with torch.no_grad():
            probs = torch.softmax(self._model(X_tensor), dim=1).cpu().numpy()[0]
        
        result = {label: 0.0 for label in SENTIMENT_LABELS}
        for idx, label in enumerate(SENTIMENT_LABELS):
            result[label] = float(probs[idx])
        return result

    def save(self, output_path: str | Path) -> None:
        """Persist model artifact to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "max_features": self.max_features,
            "min_df": self.min_df,
            "ngram_range": self.ngram_range,
            "random_state": self.random_state,
            "hidden_dim": self.hidden_dim,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "dropout": self.dropout,
            "correct_spelling": self.correct_spelling,
            "single_label": self._single_label,
        }

        if self._vectorizer is not None:
            payload["vectorizer"] = self._vectorizer
        
        if self._model is not None:
            payload["model_state_dict"] = self._model.state_dict()
            payload["input_dim"] = self._model.network[0].in_features

        joblib.dump(payload, path)

    @classmethod
    def load(cls, model_path: str | Path) -> "AspectConditionedSentimentModel":
        """Load model artifact from disk."""
        payload = joblib.load(Path(model_path))

        model = cls(
            max_features=int(payload["max_features"]),
            min_df=int(payload["min_df"]),
            ngram_range=tuple(payload["ngram_range"]),
            random_state=int(payload["random_state"]),
            hidden_dim=int(payload.get("hidden_dim", 256)),
            epochs=int(payload.get("epochs", 30)),
            batch_size=int(payload.get("batch_size", 64)),
            learning_rate=float(payload.get("learning_rate", 0.001)),
            dropout=float(payload.get("dropout", 0.3)),
            correct_spelling=bool(payload.get("correct_spelling", False)),
        )
        
        model._single_label = payload.get("single_label")
        model._vectorizer = payload.get("vectorizer")
        
        if "model_state_dict" in payload:
            model._model = model._build_model(int(payload["input_dim"]))
            model._model.load_state_dict(payload["model_state_dict"])
            model._model.eval()
        
        return model