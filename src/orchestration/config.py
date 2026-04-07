"""Orchestrator configuration (paths, thresholds, scenario)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Scenario = Literal["random_holdout", "age_shift"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class OrchestratorConfig:
    """Paths and behaviour for the drift orchestration loop."""

    root: Path = field(default_factory=project_root)
    baseline_path: Path | None = None
    sqlite_path: Path | None = None
    # How to build "current" data for monitoring (demo / thesis)
    scenario: Scenario = "random_holdout"
    test_size: float = 0.3
    random_state: int = 42
    # Policy thresholds (see DriftThresholdPolicy)
    max_high_psi_features: int = 0
    max_ks_significant_numeric: int = 2
    max_chi2_significant_categorical: int = 3

    def __post_init__(self) -> None:
        if self.baseline_path is None:
            self.baseline_path = self.root / "artifacts" / "baseline_profile.json"
        if self.sqlite_path is None:
            self.sqlite_path = self.root / "artifacts" / "orchestration.db"
