from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    train_csv: Path = Path("data/DeepX_train.csv")
    validation_csv: Path = Path("data/DeepX_validation.csv")
    unlabeled_csv: Path = Path("data/DeepX_unlabeled.csv")


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path = Path("artifacts")
    aspect_model_dir: Path = Path("artifacts/aspect_model")
    calibration_dir: Path = Path("artifacts/calibration")
    reports_dir: Path = Path("artifacts/reports")


@dataclass(frozen=True)
class AspectTrainingSettings:
    random_seed: int = 42
    max_text_features: int = 60_000
    transformer_max_vocab: int = 20_000
    transformer_max_seq_len: int = 128
    transformer_epochs: int = 4
    transformer_batch_size: int = 32
    transformer_learning_rate: float = 3e-4
    ensemble_linear_weight: float = 0.65

