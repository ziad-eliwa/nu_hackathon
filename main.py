"""Small CLI shim for common project actions.

This file remains intentionally lightweight. Core logic lives in `src/absa`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        print("NU Arabic ABSA utility CLI")
        print("Usage: main.py <command>")
        print("Commands:")
        print("  train-aspect        Train Path A aspect detection models")
        print("  train-sentiment     Train Path B aspect-conditioned sentiment model")
        print("  predict             Run end-to-end inference (Path A + Path B)")
        print("  evaluate            Evaluate predictions against validation set")
        print("  semi-supervised     Train with pseudo-labeling on unlabeled data")
        print("  run-improvements    Run preprocessing/model improvement experiments")
        return 0

    command = argv[0]

    if command == "train-aspect":
        import importlib
        train_module = importlib.import_module("absa.training.train_aspect")
        return train_module.main()

    if command == "train-sentiment":
        import importlib
        train_module = importlib.import_module("absa.training.train_sentiment")
        return train_module.main(argv[1:])

    if command == "predict":
        return _run_predict(argv[1:])

    if command == "evaluate":
        return _run_evaluate(argv[1:])

    if command == "semi-supervised":
        import importlib
        train_module = importlib.import_module("absa.training.semi_supervised")
        return train_module.main()

    if command == "run-improvements":
        import importlib
        module = importlib.import_module("absa.training.run_improvements")
        return module.main(argv[1:])

    print(f"Unknown command: {command}")
    return 1


def _run_predict(argv: list[str]) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run end-to-end inference")
    parser.add_argument("--input-csv", default="data/DeepX_validation.csv")
    parser.add_argument("--output-json", default="predictions.json")
    parser.add_argument("--model-dir", default="artifacts")
    args = parser.parse_args(argv)

    from absa.data.io import load_labeled_reviews
    from absa.data.schemas import ReviewInput, review_from_mapping
    from absa.inference.predict import ABSAPredictor, JsonAspectProbabilityProvider
    from absa.inference.submission import write_submission_json
    from absa.models.aspect_api import AspectEnsemblePredictor
    from absa.models.sentiment_transformer import AspectConditionedSentimentModel
    from absa.config.settings import InferenceSettings

    records = load_labeled_reviews(args.input_csv)
    reviews = [
        ReviewInput(
            review_id=r.review_id,
            review_text=r.review_text,
            platform=r.platform,
            business_category=r.business_category,
        )
        for r in records
    ]

    aspect_model = AspectEnsemblePredictor.load_from_artifacts(args.model_dir)
    aspect_probs = aspect_model.predict_aspect_probs(records)
    
    from collections import defaultdict
    aspect_prob_dict = {}
    for review_id, probs in aspect_probs.items():
        aspect_prob_dict[review_id] = probs
    
    class DictAspectProvider:
        def __init__(self, probs_dict):
            self._probs = probs_dict
        def predict_aspect_probs(self, reviews):
            return [self._probs.get(r.review_id, {}) for r in reviews]
    
    aspect_provider = DictAspectProvider(aspect_prob_dict)

    sentiment_model = AspectConditionedSentimentModel.load(
        f"{args.model_dir}/sentiment_model/sentiment_model.joblib"
    )
    settings = InferenceSettings.from_threshold_file(
        f"{args.model_dir}/calibration/aspect_thresholds.json"
    )

    predictor = ABSAPredictor(
        sentiment_model=sentiment_model,
        aspect_provider=aspect_provider,
        settings=settings,
    )

    predictions = predictor.predict_reviews(reviews)
    write_submission_json(predictions, args.output_json)
    print(f"Predictions written to {args.output_json}")
    return 0


def _run_evaluate(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate predictions")
    parser.add_argument("--predictions-json", default="predictions.json")
    parser.add_argument("--validation-csv", default="data/DeepX_validation.csv")
    args = parser.parse_args(argv)

    import json
    from absa.data.io import load_labeled_reviews
    from absa.data.schemas import LabeledReviewRecord, PredictionRecord
    from absa.evaluation.metrics import evaluate_predictions

    gold_records = load_labeled_reviews(args.validation_csv)
    gold = [
        PredictionRecord(
            review_id=r.review_id,
            aspects=list(r.aspects),
            aspect_sentiments=dict(r.aspect_sentiments),
        )
        for r in gold_records
    ]

    pred_data = json.loads(open(args.predictions_json).read())
    pred = [
        PredictionRecord(
            review_id=p["review_id"],
            aspects=p["aspects"],
            aspect_sentiments=p["aspect_sentiments"],
        )
        for p in pred_data
    ]

    results = evaluate_predictions(gold, pred)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())