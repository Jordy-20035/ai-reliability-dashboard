"""Orchestrator: one pipeline run = load data → drift report → policy → actions → persist."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from src.drift_detection.report import run_drift_analysis

from .actions import Action, LogAction, PipelineContext, RetrainPipelineAction
from .config import OrchestratorConfig
from .data_context import (
    fit_or_load_baseline,
    load_feature_matrix,
    split_labeled_reference_current,
    split_reference_current,
)
from .policies import DriftThresholdPolicy
from .store import RunStore


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PipelineResult:
    policy_triggered: bool
    trigger_reasons: list[str]
    summary: dict[str, Any]
    run_id: int | None = None
    scenario: str = ""


class Orchestrator:
    """
    Wires drift detection into policy evaluation and actions.

    Typical actions: log + placeholder retrain when thresholds are exceeded.
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        policy: DriftThresholdPolicy | None = None,
        actions: Sequence[Action] | None = None,
        store: RunStore | None = None,
    ) -> None:
        self.config = config
        self.policy = policy or DriftThresholdPolicy(
            max_high_psi_features=config.max_high_psi_features,
            max_ks_significant_numeric=config.max_ks_significant_numeric,
            max_chi2_significant_categorical=config.max_chi2_significant_categorical,
        )
        self.actions: list[Action] = list(actions) if actions else [LogAction(), RetrainPipelineAction()]
        self.store = store or RunStore(config.sqlite_path)  # type: ignore[arg-type]

    def run_pipeline(self) -> PipelineResult:
        started = _utc_now()
        X = load_feature_matrix()
        ref, cur = split_reference_current(
            X,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            scenario=self.config.scenario,
        )
        labeled_ref, labeled_cur = split_labeled_reference_current(
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            scenario=self.config.scenario,
        )
        assert self.config.baseline_path is not None
        baseline = fit_or_load_baseline(ref, self.config.baseline_path)

        report = run_drift_analysis(ref, cur, baseline)
        triggered, reasons = self.policy.evaluate(report)

        ctx = PipelineContext(
            report=report,
            policy_triggered=triggered,
            trigger_reasons=reasons,
            metadata={"scenario": self.config.scenario},
            labeled_reference=labeled_ref,
            labeled_current=labeled_cur,
        )
        for act in self.actions:
            act.run(ctx)

        finished = _utc_now()
        run_id = self.store.insert_run(
            scenario=self.config.scenario,
            policy_triggered=triggered,
            trigger_reasons=reasons,
            summary=report.summary,
            started_at=started,
            finished_at=finished,
        )

        return PipelineResult(
            policy_triggered=triggered,
            trigger_reasons=reasons,
            summary=report.summary,
            run_id=run_id,
            scenario=self.config.scenario,
        )
