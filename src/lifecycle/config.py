"""Lifecycle store location."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class LifecycleConfig:
    root: Path
    db_path: Path | None = None

    def __post_init__(self) -> None:
        if self.db_path is None:
            self.db_path = self.root / "artifacts" / "lifecycle.db"


def default_lifecycle_config() -> LifecycleConfig:
    return LifecycleConfig(root=project_root())
