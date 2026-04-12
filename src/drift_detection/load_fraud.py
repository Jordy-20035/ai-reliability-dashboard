"""Load credit card fraud CSV (same schema as Kaggle creditcard.csv)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema_fraud import FRAUD_FEATURE_COLS, FRAUD_TARGET_COL, FRAUD_TIME_COL


def load_fraud_csv(path: Path | str) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Fraud dataset not found: {p}")
    df = pd.read_csv(p)
    required = {FRAUD_TIME_COL, *FRAUD_FEATURE_COLS, FRAUD_TARGET_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Fraud CSV missing columns: {sorted(missing)}")
    return df.reset_index(drop=True)
