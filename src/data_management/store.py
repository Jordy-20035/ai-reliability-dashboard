"""Schema for dataset versioning, baseline artifacts, and reproducibility links."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS dataset_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    source_path TEXT,
    content_hash TEXT NOT NULL UNIQUE,
    row_count INTEGER NOT NULL,
    columns_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS baseline_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    artifact_path TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    dataset_version_id INTEGER REFERENCES dataset_versions(id),
    summary_json TEXT,
    created_at TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS training_provenance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lifecycle_experiment_id INTEGER,
    lifecycle_model_version_num INTEGER,
    dataset_version_id INTEGER NOT NULL REFERENCES dataset_versions(id),
    baseline_snapshot_id INTEGER REFERENCES baseline_snapshots(id),
    git_sha TEXT,
    extra_json TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dataset_hash ON dataset_versions(content_hash);
CREATE INDEX IF NOT EXISTS idx_baseline_hash ON baseline_snapshots(content_hash);
CREATE INDEX IF NOT EXISTS idx_prov_experiment ON training_provenance(lifecycle_experiment_id);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), default=str)


def json_loads(s: str) -> Any:
    return json.loads(s)
