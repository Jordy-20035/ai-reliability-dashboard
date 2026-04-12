"""Build reference / current frames and baseline for orchestration runs."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.drift_detection.baseline import BaselineProfile, build_baseline
from src.drift_detection.load import load_adult_csv
from src.drift_detection.load_fraud import load_fraud_csv
from src.drift_detection.schema import ADULT_CATEGORICAL_FEATURES, ADULT_NUMERIC_FEATURES
from src.drift_detection.schema_fraud import FRAUD_FEATURE_COLS

from .config import OrchestratorConfig

FEATURE_COLS = list(ADULT_NUMERIC_FEATURES) + list(ADULT_CATEGORICAL_FEATURES)

FRAUD_DRIFT_SCENARIOS = frozenset({"fraud_d1_vs_d2", "fraud_d2_vs_d3", "fraud_d1_vs_d3"})


def load_feature_matrix() -> pd.DataFrame:
    df = load_adult_csv()
    return df[FEATURE_COLS]


def split_reference_current(
    X: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
    scenario: str,
    current_csv_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    reference = training slice; current = monitoring slice.

    scenario:
      random_holdout — same population, small drift expected.
      age_shift — current = rows with age >= 40 (strong drift vs reference).
      incoming_csv — current = rows loaded from external CSV path.
    """
    ref, cur = train_test_split(X, test_size=test_size, random_state=random_state)
    if scenario == "random_holdout":
        return ref.reset_index(drop=True), cur.reset_index(drop=True)
    if scenario == "age_shift":
        full = load_adult_csv()
        mask = full["age"] >= 40
        cur_shifted = full.loc[mask, FEATURE_COLS].reset_index(drop=True)
        return ref.reset_index(drop=True), cur_shifted
    if scenario == "incoming_csv":
        if not current_csv_path:
            raise ValueError(
                "incoming_csv scenario requires current_csv_path (or ORCH_CURRENT_CSV_PATH env var)"
            )
        current_full = load_adult_csv(current_csv_path)
        return ref.reset_index(drop=True), current_full[FEATURE_COLS].reset_index(drop=True)
    raise ValueError(f"Unknown adult scenario: {scenario}")


def split_labeled_reference_current(
    *,
    test_size: float,
    random_state: int,
    scenario: str,
    current_csv_path: str | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
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
    if scenario == "incoming_csv":
        if not current_csv_path:
            raise ValueError(
                "incoming_csv scenario requires current_csv_path (or ORCH_CURRENT_CSV_PATH env var)"
            )
        ref_full, _ = train_test_split(
            full, test_size=test_size, random_state=random_state
        )
        cur_full = load_adult_csv(current_csv_path)
        if "income" not in cur_full.columns:
            return ref_full.reset_index(drop=True), None
        return ref_full.reset_index(drop=True), cur_full.reset_index(drop=True)
    raise ValueError(f"Unknown adult scenario: {scenario}")


def _require_fraud_paths(cfg: OrchestratorConfig, *keys: str) -> dict[str, str]:
    m = {"d1": cfg.fraud_d1_path, "d2": cfg.fraud_d2_path, "d3": cfg.fraud_d3_path}
    out: dict[str, str] = {}
    for k in keys:
        p = m[k]
        if not p:
            raise ValueError(
                f"Fraud scenario requires FRAUD_{k.upper()}_PATH (or query override). Missing: {k}"
            )
        out[k] = p
    return out


def fraud_feature_reference_current(cfg: OrchestratorConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Feature matrices (no Time/Class) for drift, from D1/D2/D3 CSV paths."""
    scen = cfg.scenario
    cols = list(FRAUD_FEATURE_COLS)
    if scen == "fraud_d1_vs_d2":
        paths = _require_fraud_paths(cfg, "d1", "d2")
        d1 = load_fraud_csv(paths["d1"])
        d2 = load_fraud_csv(paths["d2"])
        return d1[cols].copy(), d2[cols].copy()
    if scen == "fraud_d2_vs_d3":
        paths = _require_fraud_paths(cfg, "d2", "d3")
        d2 = load_fraud_csv(paths["d2"])
        d3 = load_fraud_csv(paths["d3"])
        return d2[cols].copy(), d3[cols].copy()
    if scen == "fraud_d1_vs_d3":
        paths = _require_fraud_paths(cfg, "d1", "d3")
        d1 = load_fraud_csv(paths["d1"])
        d3 = load_fraud_csv(paths["d3"])
        return d1[cols].copy(), d3[cols].copy()
    raise ValueError(f"Not a fraud drift scenario: {scen}")


def fraud_labeled_reference_current(cfg: OrchestratorConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full fraud rows (features + Class + Time) for retrain merge after drift trigger."""
    scen = cfg.scenario
    need = ("d1", "d2") if scen == "fraud_d1_vs_d2" else ("d2", "d3") if scen == "fraud_d2_vs_d3" else ("d1", "d3")
    paths = _require_fraud_paths(cfg, *need)
    a = load_fraud_csv(paths[need[0]])
    b = load_fraud_csv(paths[need[1]])
    return a.copy(), b.copy()


def fraud_retrain_labeled_frames(cfg: OrchestratorConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """D1 + D2 full rows for manual fraud retrain API."""
    paths = _require_fraud_paths(cfg, "d1", "d2")
    return load_fraud_csv(paths["d1"]), load_fraud_csv(paths["d2"])


def fit_or_load_baseline(
    reference: pd.DataFrame,
    path,
    *,
    psi_bins: int = 10,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    metadata_source: str = "orchestration.fit_or_load_baseline",
) -> BaselineProfile:
    """Load baseline JSON if present; otherwise fit on reference and save."""
    from pathlib import Path

    path = Path(path)
    num = numeric_features if numeric_features is not None else list(ADULT_NUMERIC_FEATURES)
    cat = categorical_features if categorical_features is not None else list(ADULT_CATEGORICAL_FEATURES)
    if path.is_file():
        loaded = BaselineProfile.load(path)
        if loaded.numeric_features != num or loaded.categorical_features != cat:
            baseline = build_baseline(
                reference,
                num,
                cat,
                psi_bins=psi_bins,
                metadata={"source": metadata_source, "rebuilt": True},
            )
            baseline.save(path)
            return baseline
        return loaded
    baseline = build_baseline(
        reference,
        num,
        cat,
        psi_bins=psi_bins,
        metadata={"source": metadata_source},
    )
    baseline.save(path)
    return baseline
