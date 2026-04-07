"""End-to-end: merge data → train → evaluate → version → optional promote."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import RetrainConfig, default_retrain_config
from .data_merge import merge_training_data
from .deploy import should_promote
from .registry import ModelRegistry, read_champion
from .train import save_model, train_and_evaluate_holdout

logger = logging.getLogger(__name__)


@dataclass
class RetrainResult:
    version: int
    artifact_path: Path
    metrics: dict[str, float]
    promoted: bool
    promote_reason: str


def run_retrain_pipeline(
    labeled_reference: pd.DataFrame,
    labeled_current: pd.DataFrame,
    *,
    cfg: RetrainConfig | None = None,
) -> RetrainResult:
    """
    Merge labeled reference + current, train, evaluate, save artifact, register, maybe promote.

    `labeled_*` must include all feature columns + `income` target.
    """
    cfg = cfg or default_retrain_config()
    cfg.ensure_dirs()

    merged = merge_training_data(labeled_reference, labeled_current)
    pipe, metrics = train_and_evaluate_holdout(
        merged,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
    )

    registry = ModelRegistry(cfg.registry_path)  # type: ignore[arg-type]
    version = registry.next_version()
    assert cfg.models_dir is not None
    artifact = cfg.models_dir / f"model_v{version}.joblib"
    save_model(pipe, artifact)

    champion = read_champion(cfg.champion_path)  # type: ignore[arg-type]
    promote, reason = should_promote(metrics, champion, cfg)

    rec = registry.register(
        artifact_path=artifact,
        metrics=metrics,
        promoted=promote,
        notes=reason,
    )

    if promote:
        from .registry import write_champion

        write_champion(cfg.champion_path, rec)  # type: ignore[arg-type]
        logger.info("Promoted model v%s as champion (%s)", rec.version, reason)
    else:
        logger.info("Trained model v%s not promoted (%s)", rec.version, reason)

    return RetrainResult(
        version=rec.version,
        artifact_path=artifact,
        metrics=metrics,
        promoted=promote,
        promote_reason=reason,
    )
