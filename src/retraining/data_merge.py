"""Merge reference + new (current) labeled rows into a single training pool."""

from __future__ import annotations

import pandas as pd

from src.drift_detection.schema import TARGET_COL


def merge_training_data(
    labeled_reference: pd.DataFrame,
    labeled_current: pd.DataFrame,
    *,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Concatenate historical (reference) and incoming (current) samples.

    Both frames must include feature columns + TARGET_COL (e.g. income).
    """
    if TARGET_COL not in labeled_reference.columns or TARGET_COL not in labeled_current.columns:
        raise ValueError(f"Labeled frames must include target column {TARGET_COL!r}")
    merged = pd.concat([labeled_reference, labeled_current], axis=0, ignore_index=True)
    if drop_duplicates:
        merged = merged.drop_duplicates().reset_index(drop=True)
    return merged
