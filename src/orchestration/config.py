"""Orchestrator configuration (paths, thresholds, scenario)."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
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
    # Phase 3 operations
    alert_webhook_url: str | None = None
    enable_auto_retrain: bool = True
    scheduler_interval_seconds: int = 300

    def __post_init__(self) -> None:
        if self.baseline_path is None:
            self.baseline_path = self.root / "artifacts" / "baseline_profile.json"
        if self.sqlite_path is None:
            self.sqlite_path = self.root / "artifacts" / "orchestration.db"
        if self.alert_webhook_url is None:
            self.alert_webhook_url = (os.getenv("ORCH_ALERT_WEBHOOK_URL") or "").strip() or None
        self.enable_auto_retrain = _env_bool("ORCH_ENABLE_AUTO_RETRAIN", self.enable_auto_retrain)
        self.scheduler_interval_seconds = _env_int(
            "ORCH_SCHEDULER_INTERVAL",
            self.scheduler_interval_seconds,
        )


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
