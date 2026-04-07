"""SQLite store for dataset versions, baseline snapshots, and training provenance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class DataManagementConfig:
    root: Path
    db_path: Path | None = None

    def __post_init__(self) -> None:
        if self.db_path is None:
            self.db_path = self.root / "artifacts" / "data_management.db"


def default_data_management_config() -> DataManagementConfig:
    return DataManagementConfig(root=project_root())
