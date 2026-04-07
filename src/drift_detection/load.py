"""Load and clean Adult Census Income from CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def default_adult_csv_path(project_root: Path | None = None) -> Path:
    """Path to data/raw/adult.csv relative to project root."""
    root = project_root or Path(__file__).resolve().parents[2]
    return root / "data" / "raw" / "adult.csv"


def load_adult_csv(
    path: Path | str | None = None,
    *,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """
    Load Adult dataset from CSV.

    UCI-style missing values are marked as '?' in the file; they are replaced
    with NaN and rows with any missing feature values can be dropped.
    """
    csv_path = Path(path) if path else default_adult_csv_path()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.replace("?", pd.NA).replace(" ?", pd.NA)

    if drop_incomplete:
        df = df.dropna(axis=0, how="any")

    return df.reset_index(drop=True)
