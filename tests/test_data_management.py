from pathlib import Path

import pandas as pd
import pytest

from src.data_management.config import DataManagementConfig
from src.data_management.hashing import dataframe_content_hash
from src.data_management.service import DataManagementService


@pytest.fixture
def dm(tmp_path: Path) -> DataManagementService:
    cfg = DataManagementConfig(root=tmp_path, db_path=tmp_path / "dm.db")
    return DataManagementService(cfg)


def test_dataframe_hash_stable(dm: DataManagementService) -> None:
    a = pd.DataFrame({"b": [2, 1], "a": [3, 4]})
    b = pd.DataFrame({"a": [4, 3], "b": [1, 2]})
    assert dataframe_content_hash(a) == dataframe_content_hash(b)


def test_register_dataset_dedupe(dm: DataManagementService) -> None:
    df = pd.DataFrame({"x": [1, 2]})
    i1 = dm.register_dataset_from_dataframe(df, name="t1")
    i2 = dm.register_dataset_from_dataframe(df, name="t2")
    assert i1 == i2


def test_provenance_row(dm: DataManagementService) -> None:
    did = dm.register_dataset_from_dataframe(pd.DataFrame({"z": [1]}), name="p")
    pid = dm.record_training_provenance(
        dataset_version_id=did,
        baseline_snapshot_id=None,
        lifecycle_experiment_id=99,
        lifecycle_model_version_num=1,
        extra={"k": "v"},
    )
    rows = dm.list_provenance(limit=5)
    assert any(r["id"] == pid for r in rows)
