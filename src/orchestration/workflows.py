"""
Workflow composition.

`Orchestrator.run_pipeline` is the default linear workflow:

1. Load feature matrix and split reference vs current (scenario-dependent).
2. Load or fit baseline profile.
3. Run drift analysis (`run_drift_analysis`).
4. Evaluate `DriftThresholdPolicy`.
5. Execute `Action` hooks (log, placeholder retrain, …).
6. Append a row to SQLite `RunStore`.

Extend by adding `Action` implementations or wrapping `Orchestrator` with extra steps
(e.g. data-quality gate before drift, notification after trigger).
"""

from __future__ import annotations

from collections.abc import Callable


def run_linear(_name: str, steps: list[Callable[[], None]]) -> None:
    """Run ordered side-effect steps (simple DAG = one list)."""
    for step in steps:
        step()
