"""Train a tabular model (sklearn pipeline) on merged Adult data."""

from __future__ import annotations

import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.drift_detection.schema import (
    ADULT_CATEGORICAL_FEATURES,
    ADULT_NUMERIC_FEATURES,
    TARGET_COL,
)
from src.drift_detection.schema_fraud import FRAUD_FEATURE_COLS, FRAUD_TARGET_COL


def build_model_pipeline(random_state: int = 42) -> Pipeline:
    """Gradient boosting on numeric + one-hot categoricals (CPU-friendly)."""
    numeric = list(ADULT_NUMERIC_FEATURES)
    categorical = list(ADULT_CATEGORICAL_FEATURES)

    pre = ColumnTransformer(
        [
            ("num", "passthrough", numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ]
    )

    clf = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=200,
        random_state=random_state,
    )
    return Pipeline([("prep", pre), ("clf", clf)])


def build_fraud_model_pipeline(random_state: int = 42) -> Pipeline:
    """HistGradientBoosting on V1–V28 + Amount (numeric only)."""
    numeric = list(FRAUD_FEATURE_COLS)
    pre = ColumnTransformer([("num", "passthrough", numeric)])
    clf = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=200,
        random_state=random_state,
    )
    return Pipeline([("prep", pre), ("clf", clf)])


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    feature_cols = list(ADULT_NUMERIC_FEATURES) + list(ADULT_CATEGORICAL_FEATURES)
    X = df[feature_cols]
    y = (df[TARGET_COL].astype(str).str.contains(">50K")).astype(int).values
    return X, y


def prepare_xy_fraud(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    X = df[list(FRAUD_FEATURE_COLS)]
    y = df[FRAUD_TARGET_COL].astype(int).to_numpy()
    return X, y


def train_and_evaluate_holdout(
    merged_labeled: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, float]]:
    """Train on train split; evaluate on holdout."""
    X, y = prepare_xy(merged_labeled)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    pipe = build_model_pipeline(random_state=random_state)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "n_train": float(len(X_train)),
        "n_test": float(len(X_test)),
    }
    return pipe, metrics


def train_and_evaluate_holdout_fraud(
    merged_labeled: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, float]]:
    X, y = prepare_xy_fraud(merged_labeled)
    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    pipe = build_fraud_model_pipeline(random_state=random_state)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "n_train": float(len(X_train)),
        "n_test": float(len(X_test)),
    }
    return pipe, metrics


def save_model(pipe: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)
