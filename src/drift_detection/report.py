"""Run drift analysis: reference vs current, using a fitted baseline profile."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .baseline import BaselineProfile
from .detectors import (
    categorical_chi_square,
    categorical_psi,
    interpret_chi2_pvalue,
    interpret_ks_pvalue,
    interpret_psi,
    numerical_ks,
    numerical_psi,
)


@dataclass
class DriftReport:
    """Per-feature drift metrics and a simple aggregate."""

    feature_results: pd.DataFrame
    summary: dict[str, Any] = field(default_factory=dict)


def run_drift_analysis(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    baseline: BaselineProfile,
    *,
    ks_alpha: float = 0.05,
    chi2_alpha: float = 0.05,
) -> DriftReport:
    """
    Compare `current` to `reference` using PSI bins frozen in `baseline`.

    Numeric: PSI (binned) + KS.
    Categorical: PSI on category proportions + chi-square homogeneity.
    """
    rows: list[dict[str, Any]] = []

    for col in baseline.numeric_features:
        edges = np.asarray(baseline.numeric_bin_edges[col], dtype=float)
        ref_s = reference[col]
        cur_s = current[col]
        psi = numerical_psi(ref_s, cur_s, edges)
        ks_stat, ks_p = numerical_ks(ref_s, cur_s)
        rows.append(
            {
                "feature": col,
                "kind": "numeric",
                "psi": psi,
                "psi_band": interpret_psi(psi),
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_p,
                "ks_interpretation": interpret_ks_pvalue(ks_p, alpha=ks_alpha),
                "chi2_statistic": np.nan,
                "chi2_pvalue": np.nan,
                "chi2_dof": np.nan,
                "chi2_interpretation": None,
            }
        )

    for col in baseline.categorical_features:
        ref_s = reference[col]
        cur_s = current[col]
        psi = categorical_psi(ref_s, cur_s)
        c2, c2_p, dof = categorical_chi_square(ref_s, cur_s)
        rows.append(
            {
                "feature": col,
                "kind": "categorical",
                "psi": psi,
                "psi_band": interpret_psi(psi),
                "ks_statistic": np.nan,
                "ks_pvalue": np.nan,
                "ks_interpretation": None,
                "chi2_statistic": c2,
                "chi2_pvalue": c2_p,
                "chi2_dof": dof,
                "chi2_interpretation": interpret_chi2_pvalue(c2_p, alpha=chi2_alpha),
            }
        )

    df = pd.DataFrame(rows)
    n_num = (df["kind"] == "numeric").sum()
    n_cat = (df["kind"] == "categorical").sum()
    high_psi = int((df["psi_band"] == "high").sum())
    ks_sig = int((df.loc[df["kind"] == "numeric", "ks_interpretation"] == "significant").sum())
    chi_sig = int(
        (df.loc[df["kind"] == "categorical", "chi2_interpretation"] == "significant").sum()
    )

    summary = {
        "n_reference_rows": len(reference),
        "n_current_rows": len(current),
        "n_numeric_features": int(n_num),
        "n_categorical_features": int(n_cat),
        "n_features_high_psi": high_psi,
        "n_numeric_ks_significant": ks_sig,
        "n_categorical_chi2_significant": chi_sig,
    }

    return DriftReport(feature_results=df, summary=summary)
