"""On-demand HTTP trigger for drift pipeline (optional real-time)."""

from __future__ import annotations

from fastapi import FastAPI

from .engine import Orchestrator


def create_app(orchestrator: Orchestrator) -> FastAPI:
    app = FastAPI(title="Trustworthy AI — Orchestration", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/run/drift-check")
    def run_drift_check() -> dict:
        result = orchestrator.run_pipeline()
        return {
            "policy_triggered": result.policy_triggered,
            "trigger_reasons": result.trigger_reasons,
            "summary": result.summary,
            "run_id": result.run_id,
            "scenario": result.scenario,
        }

    return app
