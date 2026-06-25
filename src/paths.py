"""Resolve artifact paths across local dev and Docker (/app/artifacts)."""

from __future__ import annotations

from pathlib import Path

from src.retraining.config import project_root


def resolve_artifact_path(path: str | Path) -> Path:
    """
    Normalize stored artifact paths.

    Lifecycle rows created in Docker use `/app/artifacts/...`; locally the same
    files live under `<project>/artifacts/...`.
    """
    raw = Path(path)
    if raw.is_file():
        return raw

    normalized = str(path).replace("\\", "/")
    if normalized.startswith("/app/"):
        candidate = project_root() / normalized.removeprefix("/app/")
        if candidate.is_file():
            return candidate

    if raw.name.endswith(".joblib"):
        fallback = project_root() / "artifacts" / "models" / raw.name
        if fallback.is_file():
            return fallback

    return raw
