"""Streamlit GUI for CSV-based ABSA inference and visualization."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import streamlit as st

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[2]
PROJECT_ROOT = CURRENT_FILE.parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from absa.config.settings import InferenceSettings
from absa.config.taxonomy import ASPECT_TAXONOMY, NONE_ASPECT, SENTIMENT_LABELS
from absa.data.schemas import (
    PredictionRecord,
    ReviewRecord,
    parse_aspects_raw,
    parse_aspect_sentiments_raw,
)
from absa.evaluation.metrics import evaluate_predictions
from absa.inference.predict import ABSAPredictor, reviews_from_dataframe
from absa.inference.submission import build_submission_rows
from absa.models.aspect_api import AspectEnsemblePredictor
from absa.models.sentiment_transformer import AspectConditionedSentimentModel

DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_SENTIMENT_MODEL = DEFAULT_ARTIFACT_ROOT / "sentiment_model" / "sentiment_model.joblib"
DEFAULT_THRESHOLD_FILE = DEFAULT_ARTIFACT_ROOT / "calibration" / "aspect_thresholds.json"


class DictAspectProbabilityProvider:
    """In-memory adapter that serves aspect probabilities by review_id."""

    def __init__(self, by_review_id: Mapping[str, Mapping[str, float]]) -> None:
        self._by_review_id = {
            str(review_id): {str(a): float(p) for a, p in probs.items()}
            for review_id, probs in by_review_id.items()
        }

    def predict_aspect_probs(self, reviews: Sequence[Any]) -> list[Mapping[str, float]]:
        return [self._by_review_id.get(str(review.review_id), {}) for review in reviews]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip().lower()
    return text in {"", "nan", "none", "null"}


def _safe_str(value: Any) -> str:
    if _is_missing(value):
        return ""
    return str(value).strip()


def _safe_int(value: Any, fallback: int = 0) -> int:
    if _is_missing(value):
        return fallback
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return fallback


def _ensure_required_columns(df: pd.DataFrame) -> None:
    if "review_text" not in df.columns:
        raise ValueError("Input CSV must contain a 'review_text' column")

    if "review_id" not in df.columns:
        df["review_id"] = [f"row_{idx}" for idx in range(len(df))]
    else:
        review_ids: list[str] = []
        for idx, value in enumerate(df["review_id"].tolist()):
            rid = _safe_str(value)
            review_ids.append(rid if rid else f"row_{idx}")
        df["review_id"] = review_ids

    df["review_text"] = [_safe_str(value) for value in df["review_text"].tolist()]


def _to_review_records(df: pd.DataFrame) -> list[ReviewRecord]:
    records: list[ReviewRecord] = []
    for row in df.to_dict(orient="records"):
        records.append(
            ReviewRecord(
                review_id=_safe_str(row.get("review_id")),
                review_text=_safe_str(row.get("review_text")),
                star_rating=_safe_int(row.get("star_rating"), fallback=0),
                date=_safe_str(row.get("date")),
                business_name=_safe_str(row.get("business_name")),
                business_category=_safe_str(row.get("business_category")),
                platform=_safe_str(row.get("platform")),
            )
        )
    return records


def _normalize_probabilities_payload(payload: Any) -> dict[str, dict[str, float]]:
    normalized: dict[str, dict[str, float]] = {}

    if isinstance(payload, dict):
        for review_id, probs in payload.items():
            if not isinstance(probs, Mapping):
                continue
            normalized[str(review_id)] = _normalize_single_prob_map(probs)
        return normalized

    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, Mapping):
                continue
            review_id = _safe_str(row.get("review_id"))
            probs = row.get("aspect_probs", row.get("probabilities", {}))
            if not review_id or not isinstance(probs, Mapping):
                continue
            normalized[review_id] = _normalize_single_prob_map(probs)
        return normalized

    raise ValueError(
        "Aspect probability JSON must be a dict keyed by review_id, or a list of rows"
    )


def _normalize_single_prob_map(probs: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for aspect in ASPECT_TAXONOMY:
        raw_value = probs.get(aspect, 0.0)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = 0.0
        result[aspect] = max(0.0, min(1.0, value))
    return result


@st.cache_resource(show_spinner=False)
def _load_aspect_ensemble(artifact_root: str) -> AspectEnsemblePredictor:
    return AspectEnsemblePredictor.load_from_artifacts(artifact_root)


@st.cache_resource(show_spinner=False)
def _load_sentiment_model(model_path: str) -> AspectConditionedSentimentModel:
    return AspectConditionedSentimentModel.load(model_path)


def _build_settings(
    use_threshold_file: bool,
    threshold_file: str,
    default_threshold: float,
    fallback_aspect: str,
) -> InferenceSettings:
    if use_threshold_file:
        return InferenceSettings.from_threshold_file(
            threshold_path=threshold_file,
            default_threshold=default_threshold,
            fallback_aspect=fallback_aspect,
        )

    return InferenceSettings(
        default_threshold=default_threshold,
        fallback_aspect=fallback_aspect,
    )


def _predictions_to_dataframe(predictions: Sequence[PredictionRecord]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        row: dict[str, Any] = {
            "review_id": prediction.review_id,
            "predicted_aspects": ", ".join(prediction.aspects),
            "num_predicted_aspects": len(prediction.aspects),
            "aspect_sentiments": json.dumps(
                prediction.aspect_sentiments,
                ensure_ascii=False,
            ),
        }
        for aspect in ASPECT_TAXONOMY:
            row[f"sentiment__{aspect}"] = prediction.aspect_sentiments.get(aspect, "")
        rows.append(row)
    return pd.DataFrame(rows)


def _build_aspect_counts(predictions: Sequence[PredictionRecord]) -> pd.DataFrame:
    counts = Counter()
    for prediction in predictions:
        for aspect in prediction.aspects:
            counts[aspect] += 1

    return pd.DataFrame(
        {
            "aspect": list(ASPECT_TAXONOMY),
            "count": [counts.get(aspect, 0) for aspect in ASPECT_TAXONOMY],
        }
    ).set_index("aspect")


def _build_sentiment_counts(predictions: Sequence[PredictionRecord]) -> pd.DataFrame:
    counts = Counter()
    for prediction in predictions:
        for sentiment in prediction.aspect_sentiments.values():
            counts[sentiment] += 1

    return pd.DataFrame(
        {
            "sentiment": list(SENTIMENT_LABELS),
            "count": [counts.get(sentiment, 0) for sentiment in SENTIMENT_LABELS],
        }
    ).set_index("sentiment")


def _build_aspect_sentiment_matrix(predictions: Sequence[PredictionRecord]) -> pd.DataFrame:
    matrix = pd.DataFrame(0, index=ASPECT_TAXONOMY, columns=SENTIMENT_LABELS)

    for prediction in predictions:
        for aspect, sentiment in prediction.aspect_sentiments.items():
            if aspect in matrix.index and sentiment in matrix.columns:
                matrix.loc[aspect, sentiment] += 1

    matrix.index.name = "aspect"
    return matrix


def _build_gold_records(df: pd.DataFrame) -> tuple[list[PredictionRecord], int]:
    if "aspects" not in df.columns or "aspect_sentiments" not in df.columns:
        return [], 0

    gold_records: list[PredictionRecord] = []
    skipped = 0

    for row in df.to_dict(orient="records"):
        review_id = _safe_str(row.get("review_id"))
        if not review_id:
            skipped += 1
            continue

        aspects = parse_aspects_raw(row.get("aspects"))
        sentiments = parse_aspect_sentiments_raw(row.get("aspect_sentiments"))

        if not aspects and not sentiments:
            skipped += 1
            continue

        if not aspects and sentiments:
            aspects = [a for a in ASPECT_TAXONOMY if a in sentiments]

        if NONE_ASPECT in aspects:
            aspects = [NONE_ASPECT]
            sentiments = {NONE_ASPECT: "neutral"}
        else:
            aspects = [aspect for aspect in aspects if aspect in sentiments]

        if not aspects:
            skipped += 1
            continue

        gold_records.append(
            PredictionRecord(
                review_id=review_id,
                aspects=aspects,
                aspect_sentiments={aspect: sentiments.get(aspect, "neutral") for aspect in aspects},
            )
        )

    return gold_records, skipped


def _run_prediction_pipeline(
    input_df: pd.DataFrame,
    aspect_probs_by_id: Mapping[str, Mapping[str, float]],
    sentiment_model_path: str,
    settings: InferenceSettings,
) -> list[PredictionRecord]:
    reviews = reviews_from_dataframe(input_df)
    provider = DictAspectProbabilityProvider(aspect_probs_by_id)
    sentiment_model = _load_sentiment_model(sentiment_model_path)

    predictor = ABSAPredictor(
        sentiment_model=sentiment_model,
        aspect_provider=provider,
        settings=settings,
    )
    return predictor.predict_reviews(reviews)


def main() -> None:
    st.set_page_config(page_title="NU ABSA GUI", layout="wide")
    st.title("NU ABSA Inference GUI")
    st.caption(
        "Upload a test CSV, run aspect + sentiment inference, and inspect the output visually."
    )

    with st.sidebar:
        st.header("Run Settings")
        uploaded_csv = st.file_uploader("Input CSV", type=["csv"])

        aspect_source = st.radio(
            "Aspect source",
            options=["Path A artifacts", "Uploaded probabilities JSON"],
            index=0,
        )

        artifact_root = st.text_input(
            "Artifact root",
            value=str(DEFAULT_ARTIFACT_ROOT),
            help="Used when Aspect source is Path A artifacts.",
        )

        uploaded_probs_json = st.file_uploader(
            "Aspect probabilities JSON",
            type=["json"],
            disabled=aspect_source != "Uploaded probabilities JSON",
        )

        sentiment_model_path = st.text_input(
            "Sentiment model path",
            value=str(DEFAULT_SENTIMENT_MODEL),
        )

        use_threshold_file = st.checkbox("Use calibrated threshold file", value=True)
        threshold_file_path = st.text_input(
            "Threshold file path",
            value=str(DEFAULT_THRESHOLD_FILE),
            disabled=not use_threshold_file,
        )

        default_threshold = st.slider(
            "Default threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
        )
        fallback_aspect = st.selectbox(
            "Fallback aspect",
            options=list(ASPECT_TAXONOMY),
            index=ASPECT_TAXONOMY.index("general"),
        )

        run_button = st.button(
            "Run Inference",
            type="primary",
            disabled=uploaded_csv is None,
            use_container_width=True,
        )

    if uploaded_csv is not None:
        input_df = pd.read_csv(uploaded_csv)
        st.subheader("Input Preview")
        st.dataframe(input_df.head(20), use_container_width=True)
    else:
        input_df = None

    if run_button and input_df is not None:
        try:
            _ensure_required_columns(input_df)

            if aspect_source == "Path A artifacts":
                model = _load_aspect_ensemble(artifact_root)
                review_records = _to_review_records(input_df)
                aspect_probs_by_id = model.predict_aspect_probs(review_records)
            else:
                if uploaded_probs_json is None:
                    st.error("Please upload an aspect probabilities JSON file.")
                    return
                payload = json.loads(uploaded_probs_json.read().decode("utf-8"))
                aspect_probs_by_id = _normalize_probabilities_payload(payload)

            settings = _build_settings(
                use_threshold_file=use_threshold_file,
                threshold_file=threshold_file_path,
                default_threshold=default_threshold,
                fallback_aspect=fallback_aspect,
            )

            predictions = _run_prediction_pipeline(
                input_df=input_df,
                aspect_probs_by_id=aspect_probs_by_id,
                sentiment_model_path=sentiment_model_path,
                settings=settings,
            )

            prediction_df = _predictions_to_dataframe(predictions)
            submission_rows = build_submission_rows(predictions, validate=True)

            st.session_state["absa_predictions"] = predictions
            st.session_state["absa_prediction_df"] = prediction_df
            st.session_state["absa_submission_rows"] = submission_rows
            st.session_state["absa_input_df"] = input_df.copy()

        except Exception as exc:  # noqa: BLE001
            st.error("Inference failed. See details below.")
            st.exception(exc)
            return

    predictions = st.session_state.get("absa_predictions")
    prediction_df = st.session_state.get("absa_prediction_df")
    submission_rows = st.session_state.get("absa_submission_rows")
    latest_input_df = st.session_state.get("absa_input_df")

    if predictions and prediction_df is not None and submission_rows is not None:
        st.subheader("Summary")

        none_count = sum(1 for prediction in predictions if prediction.aspects == [NONE_ASPECT])
        sentiment_total = sum(len(prediction.aspect_sentiments) for prediction in predictions)
        avg_aspects = (
            sum(len(prediction.aspects) for prediction in predictions) / len(predictions)
            if predictions
            else 0.0
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Reviews", len(predictions))
        col2.metric("None-only predictions", none_count)
        col3.metric("Total aspect sentiments", sentiment_total)
        col4.metric("Avg aspects per review", f"{avg_aspects:.2f}")

        st.subheader("Aspect Frequency")
        st.bar_chart(_build_aspect_counts(predictions), use_container_width=True)

        st.subheader("Sentiment Frequency")
        st.bar_chart(_build_sentiment_counts(predictions), use_container_width=True)

        st.subheader("Aspect x Sentiment Matrix")
        st.dataframe(
            _build_aspect_sentiment_matrix(predictions),
            use_container_width=True,
        )

        st.subheader("Per-Review Predictions")
        st.dataframe(prediction_df, use_container_width=True)

        csv_bytes = prediction_df.to_csv(index=False).encode("utf-8-sig")
        json_bytes = json.dumps(submission_rows, ensure_ascii=False, indent=2).encode("utf-8")

        dl_col1, dl_col2 = st.columns(2)
        dl_col1.download_button(
            "Download predictions CSV",
            data=csv_bytes,
            file_name="absa_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
        dl_col2.download_button(
            "Download submission JSON",
            data=json_bytes,
            file_name="absa_predictions.json",
            mime="application/json",
            use_container_width=True,
        )

        if isinstance(latest_input_df, pd.DataFrame):
            gold_records, skipped_rows = _build_gold_records(latest_input_df)
            if gold_records:
                with st.expander("Evaluation (if input has gold labels)", expanded=False):
                    metrics = evaluate_predictions(gold=gold_records, pred=predictions)

                    aspect_micro = metrics["aspect_detection"]["micro_f1"]
                    aspect_macro = metrics["aspect_detection"]["macro_f1"]
                    sentiment_macro = metrics["sentiment_given_aspect"]["macro_f1"]
                    tuple_f1 = metrics["tuple"]["f1"]

                    ev_col1, ev_col2, ev_col3, ev_col4 = st.columns(4)
                    ev_col1.metric("Aspect micro-F1", f"{aspect_micro:.4f}")
                    ev_col2.metric("Aspect macro-F1", f"{aspect_macro:.4f}")
                    ev_col3.metric("Sentiment macro-F1", f"{sentiment_macro:.4f}")
                    ev_col4.metric("Tuple F1", f"{tuple_f1:.4f}")

                    st.caption(
                        f"Gold rows used: {len(gold_records)} | skipped rows: {skipped_rows}"
                    )
                    st.json(metrics)


if __name__ == "__main__":
    main()
