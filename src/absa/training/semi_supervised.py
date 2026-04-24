from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from langdetect import detect, LangDetectException

from absa.config.settings import DataPaths, ArtifactPaths
from absa.config.taxonomy import ASPECT_TAXONOMY
from absa.data.io import load_labeled_reviews
from absa.data.schemas import LabeledReviewRecord
from absa.models.aspect_api import AspectEnsemblePredictor
from absa.models.aspect_linear import AspectLinearModel
from absa.models.aspect_transformer import AspectTransformerModel
from absa.models.sentiment_transformer import AspectConditionedSentimentModel
from absa.training.calibrate import optimize_aspect_thresholds
from absa.training.train_aspect import labels_matrix
from absa.training.train_sentiment import train_and_evaluate_sentiment
from absa.inference.predict import ABSAPredictor
from absa.inference.submission import write_submission_json
from absa.evaluation.metrics import evaluate_predictions
from absa.data.schemas import PredictionRecord, ReviewInput, ReviewRecord
from absa.config.settings import InferenceSettings


def is_arabic_or_english(text: str) -> bool:
    if not text or not text.strip():
        return False
    try:
        lang = detect(text)
        return lang in ("ar", "en")
    except LangDetectException:
        return False


def load_unlabeled_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} unlabeled rows")
    
    mask = df["review_text"].apply(is_arabic_or_english)
    filtered = df[mask].copy()
    print(f"After filtering to Arabic/English: {len(filtered)} rows")
    
    return filtered


def predict_with_confidence(
    aspect_model: AspectEnsemblePredictor,
    sentiment_model: AspectConditionedSentimentModel,
    settings: InferenceSettings,
    reviews: list[ReviewRecord],
    confidence_threshold: float = 0.85,
) -> list[dict]:
    results = []
    
    aspect_probs_dict = aspect_model.predict_aspect_probs(reviews)
    
    for review in reviews:
        aspect_prob_dict = aspect_probs_dict.get(review.review_id, {})
        prob_map = {aspect: aspect_prob_dict.get(aspect, 0.0) for aspect in ASPECT_TAXONOMY}
        
        selected_aspects = [
            aspect for aspect in ASPECT_TAXONOMY
            if prob_map[aspect] >= settings.threshold_for(aspect)
        ]
        
        if not selected_aspects:
            if prob_map.get("general", 0) >= settings.threshold_for("general"):
                selected_aspects = ["general"]
            elif prob_map.get("none", 0) >= settings.threshold_for("none"):
                selected_aspects = ["none"]
        
        if not selected_aspects:
            continue
        
        sentiments = {}
        max_aspect_prob = 0.0
        
        for aspect in selected_aspects:
            if aspect == "none":
                sentiments[aspect] = "neutral"
                max_aspect_prob = max(max_aspect_prob, prob_map.get(aspect, 0))
            else:
                sentiment = sentiment_model.predict(review_text=review.review_text, aspect=aspect)
                sentiments[aspect] = sentiment
                
                aspect_prob = prob_map.get(aspect, 0)
                sentiment_prob = sentiment_model.predict_proba(review_text=review.review_text, aspect=aspect)
                
                if sentiment in sentiment_prob:
                    combined_conf = aspect_prob * sentiment_prob[sentiment]
                    max_aspect_prob = max(max_aspect_prob, combined_conf)
                else:
                    max_aspect_prob = max(max_aspect_prob, aspect_prob)
        
        if max_aspect_prob >= confidence_threshold:
            results.append({
                "review_id": review.review_id,
                "review_text": review.review_text,
                "platform": review.platform,
                "business_category": review.business_category,
                "aspects": selected_aspects,
                "aspect_sentiments": sentiments,
                "confidence": max_aspect_prob,
            })
    
    return results


def pseudo_label_unlabeled_data(
    unlabeled_csv: str,
    confidence_threshold: float = 0.85,
) -> tuple[list[LabeledReviewRecord], dict]:
    print("Loading models for pseudo-labeling...")
    
    aspect_model = AspectEnsemblePredictor.load_from_artifacts("artifacts")
    sentiment_model = AspectConditionedSentimentModel.load("artifacts/sentiment_model/sentiment_model.joblib")
    settings = InferenceSettings.from_threshold_file("artifacts/calibration/aspect_thresholds.json")
    
    print("Loading unlabeled data...")
    df = load_unlabeled_data(unlabeled_csv)
    
    reviews = [
        ReviewInput(
            review_id=str(row["review_id"]),
            review_text=str(row["review_text"]),
            platform=str(row.get("platform", "")) if pd.notna(row.get("platform")) else None,
            business_category=str(row.get("business_category", "")) if pd.notna(row.get("business_category")) else None,
        )
        for _, row in df.iterrows()
    ]
    
    from absa.data.schemas import ReviewRecord
    review_records = [
        ReviewRecord(
            review_id=str(row["review_id"]),
            review_text=str(row["review_text"]),
            star_rating=5,
            date="",
            business_name="",
            business_category=str(row.get("business_category", "")) if pd.notna(row.get("business_category")) else "",
            platform=str(row.get("platform", "")) if pd.notna(row.get("platform")) else "",
        )
        for _, row in df.iterrows()
    ]
    
    print(f"Predicting on {len(review_records)} unlabeled reviews...")
    pseudo_labels = predict_with_confidence(
        aspect_model, sentiment_model, settings, review_records, confidence_threshold
    )
    
    print(f"Pseudo-labeled {len(pseudo_labels)} high-confidence samples (>= {confidence_threshold})")
    
    labeled_records = []
    for pl in pseudo_labels:
        record = LabeledReviewRecord(
            review_id=pl["review_id"],
            review_text=pl["review_text"],
            star_rating=5,
            date="",
            business_name="",
            business_category=pl.get("business_category", ""),
            platform=pl.get("platform", ""),
            aspects=tuple(pl["aspects"]),
            aspect_sentiments=pl["aspect_sentiments"],
        )
        labeled_records.append(record)
    
    stats = {
        "total_unlabeled": len(df),
        "pseudo_labeled": len(pseudo_labels),
        "confidence_threshold": confidence_threshold,
    }
    
    return labeled_records, stats


def retrain_with_pseudo_labels(
    train_csv: str,
    validation_csv: str,
    pseudo_records: list[LabeledReviewRecord],
) -> dict:
    print("Loading original training data...")
    train_records = load_labeled_reviews(train_csv)
    val_records = load_labeled_reviews(validation_csv)
    
    print(f"Original training: {len(train_records)} samples")
    print(f"Adding {len(pseudo_records)} pseudo-labeled samples")
    
    combined_records = train_records + pseudo_records
    print(f"Combined training: {len(combined_records)} samples")
    
    y_train = labels_matrix(combined_records)
    y_val = labels_matrix(val_records)
    
    print("Retraining aspect models...")
    linear_model = AspectLinearModel()
    linear_model.fit(combined_records, y_train)
    linear_probs = linear_model.predict_proba(val_records)
    
    transformer_config = TransformerConfig()
    transformer_model = AspectTransformerModel(config=transformer_config)
    transformer_model.fit(combined_records, y_train)
    transformer_probs = transformer_model.predict_proba(val_records)
    
    from absa.training.train_aspect import _find_best_blend_weight
    best_weight = _find_best_blend_weight(y_val, linear_probs, transformer_probs)
    blended_probs = best_weight * linear_probs + (1.0 - best_weight) * transformer_probs
    thresholds = optimize_aspect_thresholds(y_true=y_val, y_prob=blended_probs)
    
    y_pred = np.zeros_like(y_val)
    for idx, record in enumerate(val_records):
        prob_map = {aspect: float(blended_probs[idx, aspect_idx]) for aspect_idx, aspect in enumerate(ASPECT_TAXONOMY)}
        from absa.training.calibrate import apply_thresholds
        selected = apply_thresholds(prob_map=prob_map, thresholds=thresholds)
        for aspect in selected:
            y_pred[idx, ASPECT_TAXONOMY.index(aspect)] = 1
    
    eval_payload = {
        "metrics": {
            "micro_f1": float(f1_score(y_val, y_pred, average="micro", zero_division=0)),
            "macro_f1": float(f1_score(y_val, y_pred, average="macro", zero_division=0)),
        },
        "ensemble": {
            "linear_weight": best_weight,
            "transformer_weight": float(1.0 - best_weight),
        },
    }
    
    root = Path("artifacts")
    model_dir = root / "aspect_model"
    linear_model.save(model_dir / "aspect_linear.pkl")
    transformer_model.save(model_dir / "aspect_transformer.pt")
    AspectEnsemblePredictor(
        linear_model=linear_model,
        transformer_model=transformer_model,
        linear_weight=best_weight,
    ).save_config(model_dir / "ensemble_config.json")
    
    from absa.training.calibrate import save_thresholds_config
    save_thresholds_config(path=root / "calibration" / "aspect_thresholds.json", thresholds=thresholds)
    
    return eval_payload


def run_semi_supervised_training(
    train_csv: str,
    validation_csv: str,
    unlabeled_csv: str,
    confidence_threshold: float = 0.85,
) -> dict:
    pseudo_records, stats = pseudo_label_unlabeled_data(
        unlabeled_csv, confidence_threshold
    )
    
    if pseudo_records:
        aspect_metrics = retrain_with_pseudo_labels(
            train_csv, validation_csv, pseudo_records
        )
    else:
        print("No pseudo-labels generated, skipping retraining")
        aspect_metrics = {}
    
    print("\n--- Retraining sentiment model ---")
    sentiment_summary = train_and_evaluate_sentiment(
        train_csv=train_csv,
        validation_csv=validation_csv,
        output_dir="artifacts/sentiment_model",
    )
    
    return {
        "pseudo_label_stats": stats,
        "aspect_metrics": aspect_metrics,
        "sentiment_metrics": sentiment_summary,
    }


def evaluate_on_validation(validation_csv: str, output_json: str) -> dict:
    records = load_labeled_reviews(validation_csv)
    reviews = [
        ReviewInput(
            review_id=r.review_id,
            review_text=r.review_text,
            platform=r.platform,
            business_category=r.business_category,
        )
        for r in records
    ]
    
    aspect_model = AspectEnsemblePredictor.load_from_artifacts("artifacts")
    aspect_probs = aspect_model.predict_aspect_probs(records)
    
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
        "artifacts/sentiment_model/sentiment_model.joblib"
    )
    settings = InferenceSettings.from_threshold_file(
        "artifacts/calibration/aspect_thresholds.json"
    )
    
    predictor = ABSAPredictor(
        sentiment_model=sentiment_model,
        aspect_provider=aspect_provider,
        settings=settings,
    )
    
    predictions = predictor.predict_reviews(reviews)
    write_submission_json(predictions, output_json)
    
    gold = [
        PredictionRecord(
            review_id=r.review_id,
            aspects=list(r.aspects),
            aspect_sentiments=dict(r.aspect_sentiments),
        )
        for r in records
    ]
    
    results = evaluate_predictions(gold, predictions)
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semi-supervised training with pseudo-labeling")
    defaults = DataPaths()
    parser.add_argument("--train-csv", default=str(defaults.train_csv))
    parser.add_argument("--validation-csv", default=str(defaults.validation_csv))
    parser.add_argument("--unlabeled-csv", default=str(defaults.unlabeled_csv))
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument("--output-json", default="predictions_after_semisupervised.json")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args, _ = parser.parse_known_args()
    
    result = run_semi_supervised_training(
        train_csv=args.train_csv,
        validation_csv=args.validation_csv,
        unlabeled_csv=args.unlabeled_csv,
        confidence_threshold=args.confidence_threshold,
    )
    
    print("\n--- Pseudo-labeling stats ---")
    print(json.dumps(result["pseudo_label_stats"], indent=2))
    
    print("\n--- Aspect metrics after retraining ---")
    print(json.dumps(result["aspect_metrics"], indent=2))
    
    print("\n--- Sentiment metrics after retraining ---")
    print(f"Validation macro-F1: {result['sentiment_metrics']['validation']['macro_f1']:.4f}")
    
    print("\n--- Evaluating on validation set ---")
    eval_results = evaluate_on_validation(args.validation_csv, args.output_json)
    print(json.dumps(eval_results, indent=2))
    
    print(f"\nPredictions saved to {args.output_json}")


if __name__ == "__main__":
    main()


from absa.training.train_aspect import TransformerConfig