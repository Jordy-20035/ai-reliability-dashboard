"""Merge reference + new (current) labeled rows into a single training pool."""

from __future__ import annotations

import pandas as pd

from src.drift_detection.schema import TARGET_COL as ADULT_TARGET_COL


def merge_training_data(
    labeled_reference: pd.DataFrame,
    labeled_current: pd.DataFrame,
    *,
    drop_duplicates: bool = True,
    target_col: str | None = None,
) -> pd.DataFrame:
    """
    Concatenate historical (reference) and incoming (current) samples.

    Both frames must include feature columns + target (income or Class).
    """
    col = target_col or ADULT_TARGET_COL
    if col not in labeled_reference.columns or col not in labeled_current.columns:
        raise ValueError(f"Labeled frames must include target column {col!r}")
    merged = pd.concat([labeled_reference, labeled_current], axis=0, ignore_index=True)
    if drop_duplicates:
        merged = merged.drop_duplicates().reset_index(drop=True)
    return merged
