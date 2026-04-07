"""Dataset versioning, baseline snapshots, and training provenance (SQLite)."""

from .config import DataManagementConfig, default_data_management_config
from .service import (
    BaselineSnapshotRecord,
    DataManagementService,
    DatasetVersionRecord,
    default_data_management_service,
)

__all__ = [
    "DataManagementConfig",
    "default_data_management_config",
    "DataManagementService",
    "default_data_management_service",
    "DatasetVersionRecord",
    "BaselineSnapshotRecord",
]
