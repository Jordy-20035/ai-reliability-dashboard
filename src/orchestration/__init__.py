"""
Automation / orchestration: schedule drift checks, evaluate policies, trigger actions.

This is the control plane that wires drift detection into repeatable workflows.
"""

from .actions import LogAction, PipelineContext, PlaceholderRetrainAction
from .engine import Orchestrator, PipelineResult
from .policies import DriftThresholdPolicy

__all__ = [
    "Orchestrator",
    "PipelineResult",
    "DriftThresholdPolicy",
    "PipelineContext",
    "LogAction",
    "PlaceholderRetrainAction",
]
