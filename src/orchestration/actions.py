"""Actions executed when a policy fires (logging, future retrain hook, etc.)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from src.drift_detection.report import DriftReport

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Everything a downstream action might need."""

    report: DriftReport
    policy_triggered: bool
    trigger_reasons: list[str]
    metadata: dict[str, Any]


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
    """
    Stand-in for an automated retraining pipeline.

    Wire this to your trainer + model registry in a later milestone.
    """

    def run(self, ctx: PipelineContext) -> None:
        if not ctx.policy_triggered:
            return
        logger.warning(
            "[PlaceholderRetrain] Would enqueue retraining job here. Reasons: %s",
            ctx.trigger_reasons,
        )
