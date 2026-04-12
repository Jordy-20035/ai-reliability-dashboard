import pandas as pd

from src.drift_detection.schema import TARGET_COL
from src.drift_detection.schema_fraud import FRAUD_TARGET_COL
from src.retraining.config import RetrainConfig
from src.retraining.data_merge import merge_training_data
from src.retraining.deploy import should_promote
from src.retraining.registry import ModelVersionRecord


def test_merge_training_data_concat():
    a = pd.DataFrame({"age": [1, 2], "income": ["<=50K", ">50K"]})
    b = pd.DataFrame({"age": [3, 4], "income": ["<=50K", "<=50K"]})
    m = merge_training_data(a, b)
    assert len(m) == 4
    assert TARGET_COL in m.columns


def test_merge_training_data_fraud_target():
    cols = {f"V{i}": [0.0] for i in range(1, 29)} | {"Amount": [1.0], "Class": [0]}
    a = pd.DataFrame(cols)
    b = pd.DataFrame({**{f"V{i}": [0.1] for i in range(1, 29)}, "Amount": [2.0], "Class": [1]})
    m = merge_training_data(a, b, target_col=FRAUD_TARGET_COL)
    assert len(m) == 2
    assert FRAUD_TARGET_COL in m.columns


def test_should_promote_first_model():
    cfg = RetrainConfig()
    ok, reason = should_promote({"f1_macro": 0.5}, None, cfg)
    assert ok and reason == "no_existing_champion"


def test_should_promote_when_improves():
    cfg = RetrainConfig()
    champ = ModelVersionRecord(
        version=1,
        artifact_path="x",
        created_at="t",
        metrics={"f1_macro": 0.5},
        promoted=True,
    )
    ok, _ = should_promote({"f1_macro": 0.6}, champ, cfg)
    assert ok
