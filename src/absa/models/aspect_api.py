from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from absa.config.taxonomy import ASPECTS
from absa.data.schemas import ReviewRecord
from absa.models.aspect_linear import AspectLinearModel
from absa.models.aspect_transformer import AspectTransformerModel


class AspectEnsemblePredictor:
    def __init__(
        self,
        linear_model: AspectLinearModel,
        transformer_model: AspectTransformerModel,
        linear_weight: float = 0.65,
    ) -> None:
        self.linear_model = linear_model
        self.transformer_model = transformer_model
        self.linear_weight = linear_weight

    def predict_aspect_probs(self, review_batch: Sequence[ReviewRecord]) -> dict[str, dict[str, float]]:
        linear_probs = self.linear_model.predict_proba(review_batch)
        transformer_probs = self.transformer_model.predict_proba(review_batch)
        probs = self.linear_weight * linear_probs + (1.0 - self.linear_weight) * transformer_probs
        output: dict[str, dict[str, float]] = {}
        for idx, review in enumerate(review_batch):
            output[review.review_id] = {
                aspect: float(probs[idx, aspect_idx]) for aspect_idx, aspect in enumerate(ASPECTS)
            }
        return output

    def save_config(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {"linear_weight": self.linear_weight}
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    @classmethod
    def load_from_artifacts(cls, artifact_root: str | Path) -> "AspectEnsemblePredictor":
        root = Path(artifact_root)
        linear_model = AspectLinearModel.load(root / "aspect_model" / "aspect_linear.pkl")
        transformer_model = AspectTransformerModel.load(root / "aspect_model" / "aspect_transformer.pt")
        config_path = root / "aspect_model" / "ensemble_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        return cls(
            linear_model=linear_model,
            transformer_model=transformer_model,
            linear_weight=float(config.get("linear_weight", 0.65)),
        )

