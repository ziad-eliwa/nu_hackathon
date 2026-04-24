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

"""Runtime settings objects.

Path B consumes calibrated thresholds from Path A. This module provides a small
typed settings container so both paths can share a stable handoff contract.
"""


import json
from dataclasses import dataclass, field
from pathlib import Path

from .taxonomy import ASPECT_TAXONOMY

DEFAULT_NONE_THRESHOLD: float = 0.5
DEFAULT_GENERAL_THRESHOLD: float = 0.55
NEUTRAL_BOOST_THRESHOLD: float = 0.35


NEUTRAL_BOOST_THRESHOLD: float = 0.35


@dataclass(slots=True)
class InferenceSettings:
    """Configuration values used during end-to-end inference.

    Attributes:
        default_threshold: Fallback threshold for aspects missing in calibration.
        aspect_thresholds: Per-aspect thresholds from Path A calibration output.
        fallback_aspect: Used when no aspect passes threshold and `none` is weak.
        suppress_none_when_other_aspects_exist: Enforce exclusivity preference.
        fallback_none_threshold: Threshold for treating "none" as detected when no concrete aspects pass.
        fallback_general_threshold: Threshold for treating "general" as detected in fallback.
        sentiment_thresholds: Per-sentiment thresholds for calibration (e.g., lower threshold for neutral).
        platform_settings: Per-platform inference settings for platform-specific calibration.
    """

    default_threshold: float = 0.5
    aspect_thresholds: dict[str, float] = field(default_factory=dict)
    fallback_aspect: str = "general"
    suppress_none_when_other_aspects_exist: bool = True
    fallback_none_threshold: float = DEFAULT_NONE_THRESHOLD
    fallback_general_threshold: float = DEFAULT_GENERAL_THRESHOLD
    sentiment_thresholds: dict[str, float] = field(default_factory=dict)
    platform_settings: dict[str, "InferenceSettings"] = field(default_factory=dict)

    def threshold_for(self, aspect: str) -> float:
        """Return the threshold for one aspect, with a safe default fallback."""
        return float(self.aspect_thresholds.get(aspect, self.default_threshold))

    def sentiment_threshold_for(self, sentiment: str) -> float:
        """Return the threshold for one sentiment, with a safe default fallback."""
        return float(self.sentiment_thresholds.get(sentiment, 0.5))

    def for_platform(self, platform: str) -> "InferenceSettings":
        """Return settings for a specific platform, falling back to self if not configured."""
        return self.platform_settings.get(platform, self)

    @classmethod
    def from_threshold_file(
        cls,
        threshold_path: str | Path,
        default_threshold: float = 0.5,
        fallback_aspect: str = "general",
    ) -> "InferenceSettings":
        """Load settings from a JSON threshold file produced by Path A.

        Supported formats:
        1) {"service": 0.51, ...}
        2) {"thresholds": {"service": 0.51, ...}, ...}
        3) Full format with fallback_policy from save_thresholds_config
        """
        path = Path(threshold_path)
        payload = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(payload, dict) and "thresholds" in payload:
            maybe_thresholds = payload["thresholds"]
        else:
            maybe_thresholds = payload

        if not isinstance(maybe_thresholds, dict):
            raise ValueError(
                "Threshold file must contain a mapping of aspect -> threshold."
            )

        thresholds: dict[str, float] = {}
        for aspect in ASPECT_TAXONOMY:
            raw = maybe_thresholds.get(aspect)
            if raw is None:
                continue
            thresholds[aspect] = float(raw)

        fallback_policy = payload.get("fallback_policy", {})
        none_threshold = fallback_policy.get("none_threshold", DEFAULT_NONE_THRESHOLD)
        general_threshold = fallback_policy.get("general_threshold", DEFAULT_GENERAL_THRESHOLD)

        sentiment_thresholds: dict[str, float] = {}
        raw_sentiment = payload.get("sentiment_thresholds", {})
        if isinstance(raw_sentiment, dict):
            for sentiment in ("negative", "neutral", "positive"):
                if sentiment in raw_sentiment:
                    sentiment_thresholds[sentiment] = float(raw_sentiment[sentiment])

        platform_settings: dict[str, InferenceSettings] = {}
        raw_platforms = payload.get("platform_settings", {})
        if isinstance(raw_platforms, dict):
            for platform, platform_config in raw_platforms.items():
                if isinstance(platform_config, dict):
                    platform_settings[platform] = cls(
                        default_threshold=float(default_threshold),
                        aspect_thresholds=thresholds,
                        fallback_aspect=fallback_aspect,
                        fallback_none_threshold=none_threshold,
                        fallback_general_threshold=general_threshold,
                    )

        return cls(
            default_threshold=float(default_threshold),
            aspect_thresholds=thresholds,
            fallback_aspect=fallback_aspect,
            fallback_none_threshold=none_threshold,
            fallback_general_threshold=general_threshold,
            sentiment_thresholds=sentiment_thresholds,
            platform_settings=platform_settings,
        )
