"""Error analysis helpers for ABSA predictions.

These functions produce slice-level diagnostic tables and top failure examples
that are easy to inspect in notebooks or logs.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from absa.data.schemas import PredictionRecord


def _tuple_counts(gold_item: PredictionRecord, pred_item: PredictionRecord) -> tuple[int, int, int]:
    """Return tuple-level (tp, fp, fn) for one review."""
    gold_tuples = {(aspect, gold_item.aspect_sentiments[aspect]) for aspect in gold_item.aspects}
    pred_tuples = {(aspect, pred_item.aspect_sentiments[aspect]) for aspect in pred_item.aspects}

    tp = len(gold_tuples.intersection(pred_tuples))
    fp = len(pred_tuples - gold_tuples)
    fn = len(gold_tuples - pred_tuples)
    return tp, fp, fn


def _bucket_text_length(length: int) -> str:
    """Simple length buckets used for mandatory slicing."""
    if length <= 80:
        return "0-80"
    if length <= 160:
        return "81-160"
    if length <= 320:
        return "161-320"
    return "321+"


def _bucket_aspect_count(count: int) -> str:
    """Buckets for number of aspects per review in ground truth."""
    if count <= 1:
        return "1"
    if count == 2:
        return "2"
    return "3+"


def _safe_f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


def build_comparison_frame(
    reviews_df: pd.DataFrame,
    gold: Sequence[PredictionRecord],
    pred: Sequence[PredictionRecord],
) -> pd.DataFrame:
    """Create a row-level comparison frame used by all error analyses.

    Required review metadata columns:
    - review_id
    - review_text
    Optional but recommended:
    - platform
    - business_category
    """
    gold_map = {item.review_id: item for item in gold}
    pred_map = {item.review_id: item for item in pred}
    meta_map = {
        str(row["review_id"]): row
        for _, row in reviews_df.iterrows()
        if "review_id" in reviews_df.columns
    }

    common_ids = sorted(set(gold_map).intersection(pred_map))
    rows: list[dict] = []

    for review_id in common_ids:
        gold_item = gold_map[review_id]
        pred_item = pred_map[review_id]
        meta = meta_map.get(review_id, {})

        tp, fp, fn = _tuple_counts(gold_item, pred_item)
        review_text = str(meta.get("review_text", ""))
        platform = str(meta.get("platform", "unknown"))
        business_category = str(meta.get("business_category", "unknown"))

        rows.append(
            {
                "review_id": review_id,
                "review_text": review_text,
                "platform": platform,
                "business_category": business_category,
                "text_length": len(review_text),
                "text_length_bucket": _bucket_text_length(len(review_text)),
                "gold_aspect_count": len(gold_item.aspects),
                "gold_aspect_count_bucket": _bucket_aspect_count(len(gold_item.aspects)),
                "pred_aspect_count": len(pred_item.aspects),
                "tuple_tp": tp,
                "tuple_fp": fp,
                "tuple_fn": fn,
                "tuple_exact_match": int(fp == 0 and fn == 0),
                "gold_aspects": ",".join(gold_item.aspects),
                "pred_aspects": ",".join(pred_item.aspects),
            }
        )

    return pd.DataFrame(rows)


def _slice_tuple_scores(frame: pd.DataFrame, slice_col: str) -> pd.DataFrame:
    """Aggregate tuple F1 and exact-match rate over one slice column."""
    if frame.empty:
        return pd.DataFrame(
            columns=[slice_col, "support", "tuple_f1", "exact_match_rate"]
        )

    rows: list[dict] = []
    for slice_value, group in frame.groupby(slice_col, dropna=False):
        tp = int(group["tuple_tp"].sum())
        fp = int(group["tuple_fp"].sum())
        fn = int(group["tuple_fn"].sum())
        rows.append(
            {
                slice_col: slice_value,
                "support": int(len(group)),
                "tuple_f1": _safe_f1(tp, fp, fn),
                "exact_match_rate": float(group["tuple_exact_match"].mean()),
            }
        )

    return pd.DataFrame(rows).sort_values("support", ascending=False).reset_index(drop=True)


def standard_slice_reports(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Produce mandatory slice diagnostics from the architecture plan."""
    return {
        "by_platform": _slice_tuple_scores(frame, "platform"),
        "by_length_bucket": _slice_tuple_scores(frame, "text_length_bucket"),
        "by_gold_aspect_count": _slice_tuple_scores(frame, "gold_aspect_count_bucket"),
        "by_business_category": _slice_tuple_scores(frame, "business_category"),
    }


def top_failure_examples(frame: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    """Return highest-error rows for manual inspection."""
    if frame.empty:
        return frame

    copy = frame.copy()
    copy["tuple_error_size"] = copy["tuple_fp"] + copy["tuple_fn"]
    return (
        copy.sort_values(
            ["tuple_error_size", "text_length"],
            ascending=[False, False],
        )
        .head(limit)
        .reset_index(drop=True)
    )
