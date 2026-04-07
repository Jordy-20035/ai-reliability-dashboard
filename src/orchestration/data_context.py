"""Build reference / current frames and baseline for orchestration runs."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.drift_detection.baseline import BaselineProfile, build_baseline
from src.drift_detection.load import load_adult_csv
from src.drift_detection.schema import ADULT_CATEGORICAL_FEATURES, ADULT_NUMERIC_FEATURES

FEATURE_COLS = list(ADULT_NUMERIC_FEATURES) + list(ADULT_CATEGORICAL_FEATURES)


def load_feature_matrix() -> pd.DataFrame:
    df = load_adult_csv()
    return df[FEATURE_COLS]


def split_reference_current(
    X: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
    scenario: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    reference = training slice; current = monitoring slice.

    scenario:
      random_holdout — same population, small drift expected.
      age_shift — current = rows with age >= 40 (strong drift vs reference).
    """
    ref, cur = train_test_split(X, test_size=test_size, random_state=random_state)
    if scenario == "random_holdout":
        return ref.reset_index(drop=True), cur.reset_index(drop=True)
    if scenario == "age_shift":
        full = load_adult_csv()
        mask = full["age"] >= 40
        cur_shifted = full.loc[mask, FEATURE_COLS].reset_index(drop=True)
        return ref.reset_index(drop=True), cur_shifted
    raise ValueError(f"Unknown scenario: {scenario}")


def split_labeled_reference_current(
    *,
    test_size: float,
    random_state: int,
    scenario: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Same split as `split_reference_current`, but full rows including `income` target.

    Used for automated retraining (merge reference + current with labels).
    """
    full = load_adult_csv()
    if scenario == "random_holdout":
        ref_full, cur_full = train_test_split(
            full, test_size=test_size, random_state=random_state
        )
        return ref_full.reset_index(drop=True), cur_full.reset_index(drop=True)
    if scenario == "age_shift":
        X = full[FEATURE_COLS]
        ref_X, _ = train_test_split(X, test_size=test_size, random_state=random_state)
        ref_idx = ref_X.index
        ref_full = full.loc[ref_idx].reset_index(drop=True)
        mask = full["age"] >= 40
        cur_full = full.loc[mask].reset_index(drop=True)
        return ref_full, cur_full
    raise ValueError(f"Unknown scenario: {scenario}")


def fit_or_load_baseline(
    reference: pd.DataFrame,
    path,
    *,
    psi_bins: int = 10,
) -> BaselineProfile:
    """Load baseline JSON if present; otherwise fit on reference and save."""
    from pathlib import Path

    path = Path(path)
    if path.is_file():
        return BaselineProfile.load(path)
    baseline = build_baseline(
        reference,
        list(ADULT_NUMERIC_FEATURES),
        list(ADULT_CATEGORICAL_FEATURES),
        psi_bins=psi_bins,
        metadata={"source": "orchestration.fit_or_load_baseline"},
    )
    baseline.save(path)
    return baseline
