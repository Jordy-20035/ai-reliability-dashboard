"""Orchestrator configuration (paths, thresholds, scenario)."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Literal

Scenario = Literal[
    "random_holdout",
    "age_shift",
    "incoming_csv",
    "fraud_d1_vs_d2",
    "fraud_d2_vs_d3",
    "fraud_d1_vs_d3",
]

# Retraining-only (merge D1+D2 fraud CSVs); not used by drift orchestration loop.
RetrainScenario = Literal[
    "random_holdout",
    "age_shift",
    "fraud_retrain_d1_d2",
]


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
    current_csv_path: str | None = None
    test_size: float = 0.3
    random_state: int = 42
    # Policy thresholds (see DriftThresholdPolicy)
    max_high_psi_features: int = 0
    max_ks_significant_numeric: int = 2
    max_chi2_significant_categorical: int = 3
    # Phase 3 operations
    alert_webhook_url: str | None = None
    enable_auto_retrain: bool = True
    scheduler_interval_seconds: int = 300
    # Credit card fraud temporal CSVs (from notebook export or manual split)
    fraud_d1_path: str | None = None
    fraud_d2_path: str | None = None
    fraud_d3_path: str | None = None
    fraud_baseline_path: Path | None = None

    def __post_init__(self) -> None:
        if self.baseline_path is None:
            self.baseline_path = self.root / "artifacts" / "baseline_profile.json"
        if self.sqlite_path is None:
            self.sqlite_path = self.root / "artifacts" / "orchestration.db"
        if self.current_csv_path is None:
            self.current_csv_path = (os.getenv("ORCH_CURRENT_CSV_PATH") or "").strip() or None
        if self.alert_webhook_url is None:
            self.alert_webhook_url = (os.getenv("ORCH_ALERT_WEBHOOK_URL") or "").strip() or None
        self.enable_auto_retrain = _env_bool("ORCH_ENABLE_AUTO_RETRAIN", self.enable_auto_retrain)
        self.scheduler_interval_seconds = _env_int(
            "ORCH_SCHEDULER_INTERVAL",
            self.scheduler_interval_seconds,
        )
        if self.fraud_d1_path is None:
            self.fraud_d1_path = (os.getenv("FRAUD_D1_PATH") or "").strip() or None
        if self.fraud_d2_path is None:
            self.fraud_d2_path = (os.getenv("FRAUD_D2_PATH") or "").strip() or None
        if self.fraud_d3_path is None:
            self.fraud_d3_path = (os.getenv("FRAUD_D3_PATH") or "").strip() or None
        if self.fraud_baseline_path is None:
            self.fraud_baseline_path = self.root / "artifacts" / "baseline_profile_fraud.json"


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
