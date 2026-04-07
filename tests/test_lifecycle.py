from pathlib import Path

import pytest

from src.lifecycle.config import LifecycleConfig
from src.lifecycle.service import LifecycleService
from src.lifecycle.stages import DeploymentStage


@pytest.fixture
def svc(tmp_path: Path) -> LifecycleService:
    cfg = LifecycleConfig(root=tmp_path, db_path=tmp_path / "lc.db")
    return LifecycleService(cfg)


def test_experiment_and_dev_model(svc: LifecycleService) -> None:
    eid = svc.record_experiment(
        name="exp1",
        params={"lr": 0.1},
        metrics={"f1": 0.5},
        scenario="test",
    )
    mid = svc.register_model_version(
        version_num=1,
        experiment_id=eid,
        artifact_path="m.joblib",
        metrics={"f1": 0.5},
        stage=DeploymentStage.DEVELOPMENT,
    )
    models = svc.list_models()
    assert len(models) == 1
    assert models[0].id == mid
    assert models[0].stage == DeploymentStage.DEVELOPMENT


def test_promote_to_production_archives_previous(svc: LifecycleService) -> None:
    e1 = svc.record_experiment(name="a", params={}, metrics={}, scenario="s")
    m1 = svc.register_model_version(
        version_num=1,
        experiment_id=e1,
        artifact_path="/a.joblib",
        metrics={},
        stage=DeploymentStage.PRODUCTION,
    )
    svc.set_production(m1)
    e2 = svc.record_experiment(name="b", params={}, metrics={}, scenario="s")
    m2 = svc.register_model_version(
        version_num=2,
        experiment_id=e2,
        artifact_path="/b.joblib",
        metrics={},
        stage=DeploymentStage.DEVELOPMENT,
    )
    svc.promote_stage(m2, DeploymentStage.PRODUCTION)
    rows = {m.id: m.stage for m in svc.list_models()}
    assert rows[m1] == DeploymentStage.ARCHIVED
    assert rows[m2] == DeploymentStage.PRODUCTION
