"""Decide whether a new model replaces the champion (automatic deploy / promote)."""

from __future__ import annotations

from .config import RetrainConfig
from .registry import ModelVersionRecord, metric_get


def should_promote(
    new_metrics: dict[str, float],
    champion: ModelVersionRecord | None,
    cfg: RetrainConfig,
) -> tuple[bool, str]:
    """
    Promote if no champion yet, or if primary metric improves by min_f1_improvement.
    """
    primary = cfg.primary_metric
    if champion is None:
        return True, "no_existing_champion"

    old_val = metric_get(champion.metrics, primary)
    new_val = metric_get(new_metrics, primary)
    if new_val >= old_val + cfg.min_f1_improvement:
        return True, f"{primary}_improved_or_equal"
    return False, f"{primary}_below_champion"
