"""
Drift tests: PSI (binned distributions), Kolmogorov–Smirnov (numeric),
Chi-square homogeneity (categorical).

PSI: compares binned proportions between reference and current samples.
KS: two-sample test on raw numeric values (distribution shape / location).
Chi-square: test whether category frequencies differ between samples.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-6


def psi_from_bin_counts(expected: np.ndarray, actual: np.ndarray) -> float:
    """
    Population Stability Index from bin counts.

    PSI = sum ( (a_i - e_i) * ln(a_i / e_i) ) with proportions e_i, a_i.
    """
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    if e.size == 0 or a.size == 0:
        return 0.0
    e_sum = e.sum()
    a_sum = a.sum()
    if e_sum <= 0 or a_sum <= 0:
        return 0.0
    e = e / e_sum
    a = a / a_sum
    e = np.clip(e, EPS, 1.0)
    a = np.clip(a, EPS, 1.0)
    return float(np.sum((a - e) * np.log(a / e)))


def quantile_bin_edges(series: pd.Series, n_bins: int = 10) -> np.ndarray:
    """
    Bin edges from reference quantiles (equal-frequency target).
    Drops duplicate edges from discrete / low-cardinality numerics.
    """
    clean = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if clean.empty:
        return np.array([0.0, 1.0])
    q = np.linspace(0.0, 1.0, max(n_bins + 1, 2))
    edges = np.unique(np.quantile(clean, q, method="linear"))
    if len(edges) < 2:
        lo, hi = float(clean.min()), float(clean.max())
        edges = np.array([lo, hi] if lo != hi else [lo, lo + EPS])
    return edges


def numerical_psi(
    reference: pd.Series,
    current: pd.Series,
    bin_edges: np.ndarray,
) -> float:
    """PSI for one numeric feature using fixed bin edges from reference."""
    ref = pd.to_numeric(reference, errors="coerce").dropna().astype(float)
    cur = pd.to_numeric(current, errors="coerce").dropna().astype(float)
    if ref.empty or cur.empty:
        return 0.0
    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)
    return psi_from_bin_counts(ref_counts, cur_counts)


def numerical_ks(
    reference: pd.Series,
    current: pd.Series,
) -> tuple[float, float]:
    """Two-sample Kolmogorov–Smirnov statistic and two-sided p-value."""
    ref = pd.to_numeric(reference, errors="coerce").dropna().astype(float)
    cur = pd.to_numeric(current, errors="coerce").dropna().astype(float)
    if len(ref) < 2 or len(cur) < 2:
        return 0.0, 1.0
    res = stats.ks_2samp(ref.values, cur.values)
    return float(res.statistic), float(res.pvalue)


def _merge_rare_categories(
    ref: pd.Series,
    cur: pd.Series,
    min_expected: float = 5.0,
) -> tuple[pd.Series, pd.Series]:
    """Map rare categories (low reference count or unseen in ref) to '__other__'."""
    ref_counts = ref.value_counts()
    all_cats = set(ref_counts.index) | set(cur.dropna().unique())
    rare = {c for c in all_cats if ref_counts.get(c, 0) < min_expected}

    def merge(s: pd.Series) -> pd.Series:
        if not rare:
            return s
        return s.where(~s.isin(rare) | s.isna(), other="__other__")

    if not rare:
        return ref, cur
    return merge(ref), merge(cur)


def categorical_chi_square(
    reference: pd.Series,
    current: pd.Series,
    *,
    min_expected: float = 5.0,
) -> tuple[float, float, int]:
    """
    Chi-square test of homogeneity (reference vs current category counts).

    Returns (statistic, pvalue, dof). Rare categories merged to '__other__'.
    """
    r, c = _merge_rare_categories(reference.astype("string"), current.astype("string"), min_expected)
    cats = sorted(set(r.dropna().unique()) | set(c.dropna().unique()))
    if len(cats) < 2:
        return 0.0, 1.0, 0

    ref_counts = r.value_counts().reindex(cats, fill_value=0)
    cur_counts = c.value_counts().reindex(cats, fill_value=0)
    observed = np.asarray([ref_counts.values, cur_counts.values], dtype=float)
    # scipy warns if expected < 5 in any cell; merge_rare mitigates
    chi2, p, dof, _ = stats.chi2_contingency(observed)
    return float(chi2), float(p), int(dof)


def categorical_psi(
    reference: pd.Series,
    current: pd.Series,
) -> float:
    """PSI on category proportions (same formula as binned numeric PSI)."""
    r = reference.astype("string")
    c = current.astype("string")
    cats = sorted(set(r.dropna().unique()) | set(c.dropna().unique()))
    if not cats:
        return 0.0
    e = r.value_counts().reindex(cats, fill_value=0).values.astype(float)
    a = c.value_counts().reindex(cats, fill_value=0).values.astype(float)
    return psi_from_bin_counts(e, a)


def interpret_psi(psi: float) -> Literal["stable", "moderate", "high"]:
    """Common industry-style PSI bands (heuristic, not universal)."""
    if psi < 0.1:
        return "stable"
    if psi < 0.25:
        return "moderate"
    return "high"


def interpret_ks_pvalue(p: float, alpha: float = 0.05) -> Literal["no_signal", "significant"]:
    return "significant" if p < alpha else "no_signal"


def interpret_chi2_pvalue(p: float, alpha: float = 0.05) -> Literal["no_signal", "significant"]:
    return "significant" if p < alpha else "no_signal"
