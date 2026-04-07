"""Paths and promotion rules for automated retraining."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class RetrainConfig:
    """Model artifacts live under artifacts/models/; registry is JSON."""

    root: Path = field(default_factory=project_root)
    models_dir: Path | None = None
    registry_path: Path | None = None
    champion_path: Path | None = None
    test_size: float = 0.2
    random_state: int = 42
    # Promote new model only if primary metric improves by at least this margin (tie-breaker)
    min_f1_improvement: float = 0.0
    primary_metric: str = "f1_macro"

    def __post_init__(self) -> None:
        if self.models_dir is None:
            self.models_dir = self.root / "artifacts" / "models"
        if self.registry_path is None:
            self.registry_path = self.models_dir / "registry.json"
        if self.champion_path is None:
            self.champion_path = self.models_dir / "champion.json"

    def ensure_dirs(self) -> None:
        assert self.models_dir is not None
        self.models_dir.mkdir(parents=True, exist_ok=True)


def default_retrain_config() -> RetrainConfig:
    return RetrainConfig()
