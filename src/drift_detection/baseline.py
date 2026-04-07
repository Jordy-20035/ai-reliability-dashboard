"""
Baseline profile: PSI bin edges fit on reference data (frozen for comparisons).

Persist to JSON so monitoring jobs can reload the same bins without re-reading training.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .detectors import quantile_bin_edges


@dataclass
class BaselineProfile:
    """Frozen reference summary for drift scoring."""

    numeric_features: list[str]
    categorical_features: list[str]
    psi_bins: int
    # feature_name -> monotonic bin edges (from reference quantiles)
    numeric_bin_edges: dict[str, list[float]]
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BaselineProfile:
        return cls(
            numeric_features=list(d["numeric_features"]),
            categorical_features=list(d["categorical_features"]),
            psi_bins=int(d["psi_bins"]),
            numeric_bin_edges={k: list(v) for k, v in d["numeric_bin_edges"].items()},
            metadata=d.get("metadata"),
        )

    @classmethod
    def load(cls, path: Path | str) -> BaselineProfile:
        with Path(path).open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def fit(
        cls,
        reference: pd.DataFrame,
        numeric_features: list[str],
        categorical_features: list[str],
        *,
        psi_bins: int = 10,
        metadata: dict[str, Any] | None = None,
    ) -> BaselineProfile:
        """Compute quantile bin edges on the reference (training) sample."""
        edges: dict[str, list[float]] = {}
        for col in numeric_features:
            if col not in reference.columns:
                raise KeyError(f"Missing numeric column: {col}")
            e = quantile_bin_edges(reference[col], n_bins=psi_bins)
            edges[col] = [float(x) for x in np.asarray(e).tolist()]
        for col in categorical_features:
            if col not in reference.columns:
                raise KeyError(f"Missing categorical column: {col}")
        return cls(
            numeric_features=list(numeric_features),
            categorical_features=list(categorical_features),
            psi_bins=psi_bins,
            numeric_bin_edges=edges,
            metadata=metadata,
        )


def build_baseline(
    reference: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    psi_bins: int = 10,
    metadata: dict[str, Any] | None = None,
) -> BaselineProfile:
    """Convenience wrapper around BaselineProfile.fit."""
    return BaselineProfile.fit(
        reference,
        numeric_features,
        categorical_features,
        psi_bins=psi_bins,
        metadata=metadata,
    )
