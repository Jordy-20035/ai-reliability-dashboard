"""Feature schema for Kaggle Credit Card Fraud CSV (creditcard.csv)."""

from __future__ import annotations

# All model / drift features (numeric). Time excluded from model features (ordering only).
FRAUD_FEATURE_COLS: tuple[str, ...] = tuple([f"V{i}" for i in range(1, 29)]) + ("Amount",)
FRAUD_TARGET_COL = "Class"
FRAUD_TIME_COL = "Time"
