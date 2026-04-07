"""Register datasets, baseline JSON artifacts, and training provenance rows."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DataManagementConfig, default_data_management_config
from .hashing import dataframe_content_hash, file_content_hash
from .store import connect, init_db, json_dumps, json_loads
from .summaries import distribution_summary


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
class DatasetVersionRecord:
    id: int
    name: str
    kind: str
    source_path: str | None
    content_hash: str
    row_count: int
    columns: list[str]
    created_at: str
    notes: str | None


@dataclass
class BaselineSnapshotRecord:
    id: int
    name: str | None
    artifact_path: str
    content_hash: str
    dataset_version_id: int | None
    summary: dict[str, Any] | None
    created_at: str
    notes: str | None


class DataManagementService:
    def __init__(self, cfg: DataManagementConfig | None = None) -> None:
        self.cfg = cfg or default_data_management_config()
        assert self.cfg.db_path is not None
        init_db(self.cfg.db_path)

    def _conn(self):
        return connect(self.cfg.db_path)

    def register_dataset_from_file(
        self,
        path: Path | str,
        *,
        name: str,
        notes: str | None = None,
    ) -> int:
        path = Path(path)
        h = file_content_hash(path)
        df = pd.read_csv(path)
        cols = df.columns.tolist()
        return self._insert_dataset_if_new(
            name=name,
            kind="file",
            source_path=path.as_posix(),
            content_hash=h,
            row_count=len(df),
            columns=cols,
            notes=notes,
        )

    def register_dataset_from_dataframe(
        self,
        df: pd.DataFrame,
        *,
        name: str,
        kind: str = "dataframe",
        notes: str | None = None,
    ) -> int:
        h = dataframe_content_hash(df)
        cols = df.columns.tolist()
        return self._insert_dataset_if_new(
            name=name,
            kind=kind,
            source_path=None,
            content_hash=h,
            row_count=len(df),
            columns=cols,
            notes=notes,
        )

    def _insert_dataset_if_new(
        self,
        *,
        name: str,
        kind: str,
        source_path: str | None,
        content_hash: str,
        row_count: int,
        columns: list[str],
        notes: str | None,
    ) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id FROM dataset_versions WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
            if row:
                return int(row["id"])
            cur = conn.execute(
                """
                INSERT INTO dataset_versions (name, kind, source_path, content_hash, row_count, columns_json, created_at, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    kind,
                    source_path,
                    content_hash,
                    row_count,
                    json_dumps(columns),
                    _utc_iso(),
                    notes,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def ensure_baseline_snapshot(
        self,
        artifact_path: Path | str,
        *,
        name: str | None = None,
        dataset_version_id: int | None = None,
        summary: dict[str, Any] | None = None,
        notes: str | None = None,
    ) -> int:
        """Register baseline JSON if missing; return snapshot id (dedupe by file hash)."""
        path = Path(artifact_path)
        if not path.is_file():
            raise FileNotFoundError(f"Baseline artifact not found: {path}")
        h = file_content_hash(path)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id FROM baseline_snapshots WHERE content_hash = ?",
                (h,),
            ).fetchone()
            if row:
                return int(row["id"])
            cur = conn.execute(
                """
                INSERT INTO baseline_snapshots (name, artifact_path, content_hash, dataset_version_id, summary_json, created_at, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name or path.name,
                    path.as_posix(),
                    h,
                    dataset_version_id,
                    json_dumps(summary) if summary is not None else None,
                    _utc_iso(),
                    notes,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def record_training_provenance(
        self,
        *,
        dataset_version_id: int,
        baseline_snapshot_id: int | None = None,
        lifecycle_experiment_id: int | None = None,
        lifecycle_model_version_num: int | None = None,
        git_sha: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> int:
        git_sha = git_sha or _try_git_sha()
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO training_provenance (
                    lifecycle_experiment_id, lifecycle_model_version_num,
                    dataset_version_id, baseline_snapshot_id, git_sha, extra_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lifecycle_experiment_id,
                    lifecycle_model_version_num,
                    dataset_version_id,
                    baseline_snapshot_id,
                    git_sha,
                    json_dumps(extra or {}),
                    _utc_iso(),
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list_datasets(self, limit: int = 50) -> list[DatasetVersionRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM dataset_versions ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_dataset(r) for r in rows]

    def list_baselines(self, limit: int = 50) -> list[BaselineSnapshotRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM baseline_snapshots ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_baseline(r) for r in rows]

    def list_provenance(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM training_provenance ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "id": r["id"],
                    "lifecycle_experiment_id": r["lifecycle_experiment_id"],
                    "lifecycle_model_version_num": r["lifecycle_model_version_num"],
                    "dataset_version_id": r["dataset_version_id"],
                    "baseline_snapshot_id": r["baseline_snapshot_id"],
                    "git_sha": r["git_sha"],
                    "extra": json_loads(r["extra_json"]) if r["extra_json"] else {},
                    "created_at": r["created_at"],
                }
            )
        return out

    def _row_dataset(self, r: Any) -> DatasetVersionRecord:
        return DatasetVersionRecord(
            id=r["id"],
            name=r["name"],
            kind=r["kind"],
            source_path=r["source_path"],
            content_hash=r["content_hash"],
            row_count=r["row_count"],
            columns=json_loads(r["columns_json"]),
            created_at=r["created_at"],
            notes=r["notes"],
        )

    def _row_baseline(self, r: Any) -> BaselineSnapshotRecord:
        summ = json_loads(r["summary_json"]) if r["summary_json"] else None
        return BaselineSnapshotRecord(
            id=r["id"],
            name=r["name"],
            artifact_path=r["artifact_path"],
            content_hash=r["content_hash"],
            dataset_version_id=r["dataset_version_id"],
            summary=summ if isinstance(summ, dict) else None,
            created_at=r["created_at"],
            notes=r["notes"],
        )


def default_data_management_service() -> DataManagementService:
    return DataManagementService()
