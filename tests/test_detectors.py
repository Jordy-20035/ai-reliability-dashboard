import numpy as np
import pandas as pd
import pytest

from src.drift_detection.detectors import (
    categorical_chi_square,
    numerical_psi,
    psi_from_bin_counts,
    quantile_bin_edges,
)


def test_psi_identical_distributions_near_zero():
    counts = np.array([100, 200, 300, 400], dtype=float)
    assert psi_from_bin_counts(counts, counts) < 1e-9


def test_psi_shifted_bins_positive():
    e = np.array([1000, 0, 0, 0], dtype=float)
    a = np.array([0, 0, 0, 1000], dtype=float)
    assert psi_from_bin_counts(e, a) > 0.1


def test_quantile_edges_unique():
    s = pd.Series(np.random.default_rng(0).normal(size=500))
    edges = quantile_bin_edges(s, n_bins=10)
    assert len(edges) >= 2
    assert np.all(np.diff(edges) >= 0)


def test_numerical_psi_same_series_zero():
    s = pd.Series(np.linspace(0, 1, 500))
    edges = quantile_bin_edges(s, n_bins=10)
    assert numerical_psi(s, s, edges) < 1e-6


def test_categorical_chi_square_identical():
    r = pd.Series(["a", "b", "a", "b"] * 25)
    c = pd.Series(["a", "b", "a", "b"] * 25)
    stat, p, dof = categorical_chi_square(r, c)
    assert dof >= 1
    assert p > 0.05 or stat == pytest.approx(0.0, abs=1e-6)
