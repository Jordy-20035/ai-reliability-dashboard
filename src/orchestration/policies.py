"""Decide whether drift is severe enough to trigger downstream actions."""

from __future__ import annotations

from dataclasses import dataclass

from src.drift_detection.report import DriftReport


@dataclass
class DriftThresholdPolicy:
    """
    Simple rule: fire if any count exceeds its limit.

    Tweak these for your thesis / environment; they are heuristics, not universal truths.
    """

    max_high_psi_features: int = 0
    max_ks_significant_numeric: int = 2
    max_chi2_significant_categorical: int = 3

    def evaluate(self, report: DriftReport) -> tuple[bool, list[str]]:
        """Return (should_trigger, human-readable reasons)."""
        s = report.summary
        reasons: list[str] = []

        if s.get("n_features_high_psi", 0) > self.max_high_psi_features:
            reasons.append(
                f"high_psi count {s['n_features_high_psi']} > {self.max_high_psi_features}"
            )
        if s.get("n_numeric_ks_significant", 0) > self.max_ks_significant_numeric:
            reasons.append(
                f"KS significant {s['n_numeric_ks_significant']} > {self.max_ks_significant_numeric}"
            )
        if s.get("n_categorical_chi2_significant", 0) > self.max_chi2_significant_categorical:
            reasons.append(
                f"chi2 significant {s['n_categorical_chi2_significant']} > {self.max_chi2_significant_categorical}"
            )

        return (len(reasons) > 0, reasons)
