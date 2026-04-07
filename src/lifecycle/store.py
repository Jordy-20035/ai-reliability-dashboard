"""SQLite persistence for experiments and model versions."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    params_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    scenario TEXT,
    notes TEXT,
    git_sha TEXT
);

CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version_num INTEGER NOT NULL UNIQUE,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    artifact_path TEXT NOT NULL,
    stage TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_models_stage ON model_versions(stage);
CREATE INDEX IF NOT EXISTS idx_models_version ON model_versions(version_num);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), default=str)


def json_loads(s: str) -> Any:
    return json.loads(s)
