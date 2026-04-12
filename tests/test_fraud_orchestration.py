"""Drift orchestration on minimal fraud temporal CSVs."""

from __future__ import annotations

import pandas as pd

from src.orchestration.config import OrchestratorConfig
from src.orchestration.engine import Orchestrator


def _fraud_frame(n: int, offset: float) -> pd.DataFrame:
    rows = []
    for i in range(n):
        row: dict[str, float | int] = {"Time": float(i + offset), "Amount": 1.0 + offset, "Class": i % 2}
        for j in range(1, 29):
            row[f"V{j}"] = offset + 0.01 * j + 0.001 * i
        rows.append(row)
    return pd.DataFrame(rows)


def test_fraud_d1_vs_d2_orchestration(tmp_path) -> None:
    d1 = tmp_path / "D1.csv"
    d2 = tmp_path / "D2.csv"
    _fraud_frame(12, 0.0).to_csv(d1, index=False)
    _fraud_frame(12, 2.5).to_csv(d2, index=False)

    cfg = OrchestratorConfig(
        scenario="fraud_d1_vs_d2",
        fraud_d1_path=str(d1),
        fraud_d2_path=str(d2),
        fraud_baseline_path=tmp_path / "baseline_profile_fraud.json",
        baseline_path=tmp_path / "baseline_profile_adult.json",
        sqlite_path=tmp_path / "orchestration.db",
        enable_auto_retrain=False,
    )
    orch = Orchestrator(cfg, actions=[])
    result = orch.run_pipeline()
    assert result.scenario == "fraud_d1_vs_d2"
    assert result.summary is not None
    assert result.run_id is not None
