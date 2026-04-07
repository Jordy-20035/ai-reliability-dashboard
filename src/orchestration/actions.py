"""Actions executed when a policy fires (logging, future retrain hook, etc.)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd

from src.drift_detection.report import DriftReport

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Everything a downstream action might need."""

    report: DriftReport
    policy_triggered: bool
    trigger_reasons: list[str]
    metadata: dict[str, Any]
    # Labeled rows for retraining (reference + current); same split as drift features
    labeled_reference: pd.DataFrame | None = None
    labeled_current: pd.DataFrame | None = None


class Action(Protocol):
    def run(self, ctx: PipelineContext) -> None: ...


class LogAction:
    """Emit structured log lines (works well with log aggregation later)."""

    def run(self, ctx: PipelineContext) -> None:
        if ctx.policy_triggered:
            logger.warning(
                "Drift policy TRIGGERED: %s | summary=%s",
                "; ".join(ctx.trigger_reasons),
                ctx.report.summary,
            )
        else:
            logger.info("Drift policy OK | summary=%s", ctx.report.summary)


class PlaceholderRetrainAction:
    """Log-only stub if you disable real retraining."""

    def run(self, ctx: PipelineContext) -> None:
        if not ctx.policy_triggered:
            return
        logger.warning(
            "[PlaceholderRetrain] Skipped (use RetrainPipelineAction). Reasons: %s",
            ctx.trigger_reasons,
        )


class RetrainPipelineAction:
    """
    Merge reference + current labeled data, train, evaluate, version, promote champion.

    Requires `labeled_reference` and `labeled_current` on the context.
    """

    def run(self, ctx: PipelineContext) -> None:
        if not ctx.policy_triggered:
            return
        if ctx.labeled_reference is None or ctx.labeled_current is None:
            logger.warning("Retrain skipped: missing labeled_reference / labeled_current")
            return
        from src.retraining.pipeline import run_retrain_pipeline

        result = run_retrain_pipeline(
            ctx.labeled_reference,
            ctx.labeled_current,
            scenario=str(ctx.metadata.get("scenario", "retrain")),
        )
        logger.warning(
            "Retrain finished v%s promoted=%s metrics=%s (%s)",
            result.version,
            result.promoted,
            result.metrics,
            result.promote_reason,
        )
