from __future__ import annotations

from collections import Counter

from sklearn.model_selection import train_test_split

from absa.data.schemas import LabeledReviewRecord


def multilabel_train_val_split(
    records: list[LabeledReviewRecord],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[LabeledReviewRecord], list[LabeledReviewRecord]]:
    if len(records) < 2:
        raise ValueError("Need at least 2 records to split")
    signatures = ["|".join(record.aspects) for record in records]
    signature_counts = Counter(signatures)
    can_stratify = all(count >= 2 for count in signature_counts.values())
    if can_stratify:
        train_idx, val_idx = train_test_split(
            range(len(records)),
            test_size=test_size,
            random_state=random_state,
            stratify=signatures,
        )
    else:
        train_idx, val_idx = train_test_split(
            range(len(records)),
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )
    train_records = [records[idx] for idx in train_idx]
    val_records = [records[idx] for idx in val_idx]
    return train_records, val_records

