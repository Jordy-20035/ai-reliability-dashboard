"""Content fingerprints for files and DataFrames (reproducibility)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_content_hash(path: Path) -> str:
    """SHA-256 of raw file bytes."""
    return sha256_bytes(path.read_bytes())


def dataframe_content_hash(df: pd.DataFrame) -> str:
    """
    Deterministic hash: sorted columns, lexicographic row sort, UTF-8 CSV without index.

    Same logical table (same rows/columns/values) yields the same hash even if
    row order in memory differed before sorting.
    """
    cols = sorted(df.columns)
    work = df[cols].copy()
    work = work.sort_values(by=list(cols)).reset_index(drop=True)
    payload = work.to_csv(index=False).encode("utf-8")
    return sha256_bytes(payload)
