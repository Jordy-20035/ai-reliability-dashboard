"""Model lifecycle API: experiments, versions, stages, production pointer."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
import sqlite3
from pathlib import Path
from typing import Any

from .config import LifecycleConfig, default_lifecycle_config
from .stages import STAGE_TRANSITIONS, DeploymentStage
from .store import connect, init_db, json_dumps, json_loads


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _try_git_sha() -> str | None:
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


@dataclass
class ExperimentRecord:
    id: int
    name: str
    created_at: str
    params: dict[str, Any]
    metrics: dict[str, Any]
    scenario: str | None
    notes: str | None
    git_sha: str | None


@dataclass
class ModelRecord:
    id: int
    version_num: int
    experiment_id: int
    artifact_path: str
    stage: DeploymentStage
    created_at: str
    metrics: dict[str, Any]
    notes: str | None


class LifecycleService:
    """Central lifecycle store (experiments + model versions + stages)."""

    PRODUCTION_KEY = "production_model_id"

    def __init__(self, cfg: LifecycleConfig | None = None) -> None:
        self.cfg = cfg or default_lifecycle_config()
        assert self.cfg.db_path is not None
        init_db(self.cfg.db_path)

    def _conn(self):
        return connect(self.cfg.db_path)

    def record_experiment(
        self,
        *,
        name: str,
        params: dict[str, Any],
        metrics: dict[str, Any],
        scenario: str | None = None,
        notes: str | None = None,
        git_sha: str | None = None,
    ) -> int:
        git_sha = git_sha or _try_git_sha()
        created = _utc_iso()
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO experiments (name, created_at, params_json, metrics_json, scenario, notes, git_sha)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    created,
                    json_dumps(params),
                    json_dumps(metrics),
                    scenario,
                    notes,
                    git_sha,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def register_model_version(
        self,
        *,
        version_num: int,
        experiment_id: int,
        artifact_path: Path | str,
        metrics: dict[str, Any],
        stage: DeploymentStage,
        notes: str | None = None,
    ) -> int:
        ap = Path(artifact_path).as_posix()
        created = _utc_iso()
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO model_versions (version_num, experiment_id, artifact_path, stage, created_at, metrics_json, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version_num,
                    experiment_id,
                    ap,
                    stage.value,
                    created,
                    json_dumps(metrics),
                    notes,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def set_production(self, model_row_id: int) -> None:
        """Mark model as production; archive previous production if any."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM settings WHERE key = ?",
                (self.PRODUCTION_KEY,),
            ).fetchone()
            prev = int(row["value"]) if row else None

            if prev is not None and prev != model_row_id:
                conn.execute(
                    "UPDATE model_versions SET stage = ? WHERE id = ?",
                    (DeploymentStage.ARCHIVED.value, prev),
                )

            conn.execute(
                "UPDATE model_versions SET stage = ? WHERE id = ?",
                (DeploymentStage.PRODUCTION.value, model_row_id),
            )
            conn.execute(
                """
                INSERT INTO settings(key, value) VALUES(?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (self.PRODUCTION_KEY, str(model_row_id)),
            )
            conn.commit()

    def promote_stage(self, model_row_id: int, to_stage: DeploymentStage) -> None:
        """Validate transition and update stage (manual promotion)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, stage FROM model_versions WHERE id = ?",
                (model_row_id,),
            ).fetchone()
            if not row:
                raise ValueError(f"Unknown model id {model_row_id}")
            current = DeploymentStage(row["stage"])
            if to_stage not in STAGE_TRANSITIONS.get(current, ()):
                raise ValueError(f"Cannot promote {current.value} -> {to_stage.value}")
        if to_stage == DeploymentStage.PRODUCTION:
            self.set_production(model_row_id)
            return
        with self._conn() as conn:
            conn.execute(
                "UPDATE model_versions SET stage = ? WHERE id = ?",
                (to_stage.value, model_row_id),
            )
            conn.commit()

    def list_models(self, stage: DeploymentStage | None = None) -> list[ModelRecord]:
        with self._conn() as conn:
            if stage:
                rows = conn.execute(
                    "SELECT * FROM model_versions WHERE stage = ? ORDER BY version_num DESC",
                    (stage.value,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM model_versions ORDER BY version_num DESC"
                ).fetchall()
        return [self._row_to_model(r) for r in rows]

    def list_experiments(self, limit: int = 50) -> list[ExperimentRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM experiments ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_experiment(r) for r in rows]

    def get_production_model_id(self) -> int | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM settings WHERE key = ?",
                (self.PRODUCTION_KEY,),
            ).fetchone()
        return int(row["value"]) if row else None

    def sync_from_retrain(
        self,
        *,
        experiment_name: str,
        version_num: int,
        artifact_path: Path,
        metrics: dict[str, Any],
        promoted_to_production: bool,
        notes: str,
        experiment_params: dict[str, Any],
        scenario: str,
    ) -> tuple[int, int]:
        """
        Record one training run as an experiment + model version.

        If promoted_to_production, set this model as production and archive the previous one.
        Returns (experiment_id, model_row_id).
        """
        exp_id = self.record_experiment(
            name=experiment_name,
            params=experiment_params,
            metrics=metrics,
            scenario=scenario,
            notes=notes,
        )
        stage = (
            DeploymentStage.PRODUCTION
            if promoted_to_production
            else DeploymentStage.DEVELOPMENT
        )
        mid = self.register_model_version(
            version_num=version_num,
            experiment_id=exp_id,
            artifact_path=artifact_path,
            metrics=metrics,
            stage=stage,
            notes=notes,
        )
        if promoted_to_production:
            self.set_production(mid)
        return exp_id, mid

    def _row_to_model(self, row: sqlite3.Row) -> ModelRecord:
        return ModelRecord(
            id=row["id"],
            version_num=row["version_num"],
            experiment_id=row["experiment_id"],
            artifact_path=row["artifact_path"],
            stage=DeploymentStage(row["stage"]),
            created_at=row["created_at"],
            metrics=json_loads(row["metrics_json"]),
            notes=row["notes"],
        )

    def _row_to_experiment(self, row: Any) -> ExperimentRecord:
        return ExperimentRecord(
            id=row["id"],
            name=row["name"],
            created_at=row["created_at"],
            params=json_loads(row["params_json"]),
            metrics=json_loads(row["metrics_json"]),
            scenario=row["scenario"],
            notes=row["notes"],
            git_sha=row["git_sha"],
        )


def default_lifecycle_service() -> LifecycleService:
    return LifecycleService()
