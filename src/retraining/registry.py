"""Model versioning: registry JSON + champion pointer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ModelVersionRecord:
    version: int
    artifact_path: str
    created_at: str
    metrics: dict[str, float]
    promoted: bool
    notes: str = ""


class ModelRegistry:
    def __init__(self, registry_path: Path) -> None:
        self.registry_path = registry_path
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, Any]:
        if not self.registry_path.is_file():
            return {"versions": []}
        with self.registry_path.open(encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data: dict[str, Any]) -> None:
        with self.registry_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def next_version(self) -> int:
        data = self._load()
        versions = data.get("versions", [])
        if not versions:
            return 1
        return max(v["version"] for v in versions) + 1

    def register(
        self,
        *,
        artifact_path: Path,
        metrics: dict[str, float],
        promoted: bool,
        notes: str = "",
    ) -> ModelVersionRecord:
        data = self._load()
        version = self.next_version()
        rec = ModelVersionRecord(
            version=version,
            artifact_path=str(artifact_path.as_posix()),
            created_at=_utc_iso(),
            metrics=metrics,
            promoted=promoted,
            notes=notes,
        )
        data.setdefault("versions", []).append(asdict(rec))
        self._save(data)
        return rec


def write_champion(champion_path: Path, record: ModelVersionRecord) -> None:
    champion_path.parent.mkdir(parents=True, exist_ok=True)
    with champion_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(record), f, indent=2)


def read_champion(champion_path: Path) -> ModelVersionRecord | None:
    if not champion_path.is_file():
        return None
    with champion_path.open(encoding="utf-8") as f:
        d = json.load(f)
    return ModelVersionRecord(**d)


def metric_get(metrics: dict[str, float], name: str) -> float:
    return float(metrics.get(name, 0.0))
