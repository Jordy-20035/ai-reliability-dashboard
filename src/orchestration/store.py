"""Persist orchestration run history (SQLite)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class RunRecord:
    id: int
    started_at: str
    finished_at: str
    scenario: str
    policy_triggered: bool
    trigger_reasons: list[str]
    summary: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL,
                    scenario TEXT NOT NULL,
                    policy_triggered INTEGER NOT NULL,
                    trigger_reasons TEXT NOT NULL,
                    summary_json TEXT NOT NULL
                )
                """
            )

    def insert_run(
        self,
        *,
        scenario: str,
        policy_triggered: bool,
        trigger_reasons: list[str],
        summary: dict[str, Any],
        started_at: str,
        finished_at: str,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO runs (started_at, finished_at, scenario, policy_triggered, trigger_reasons, summary_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    started_at,
                    finished_at,
                    scenario,
                    1 if policy_triggered else 0,
                    json.dumps(trigger_reasons),
                    json.dumps(summary),
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def recent(self, limit: int = 20) -> list[RunRecord]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, started_at, finished_at, scenario, policy_triggered, trigger_reasons, summary_json FROM runs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out: list[RunRecord] = []
        for r in rows:
            out.append(
                RunRecord(
                    id=r["id"],
                    started_at=r["started_at"],
                    finished_at=r["finished_at"],
                    scenario=r["scenario"],
                    policy_triggered=bool(r["policy_triggered"]),
                    trigger_reasons=json.loads(r["trigger_reasons"]),
                    summary=json.loads(r["summary_json"]),
                )
            )
        return out
