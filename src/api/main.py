"""FastAPI app that exposes orchestration, retraining, lifecycle, and data APIs."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.data_management.service import default_data_management_service
from src.lifecycle.service import default_lifecycle_service
from src.lifecycle.stages import DeploymentStage
from src.orchestration.config import OrchestratorConfig
from src.orchestration.data_context import split_labeled_reference_current
from src.orchestration.engine import Orchestrator
from src.orchestration.store import RunStore
from src.retraining.pipeline import run_retrain_pipeline


class PromoteRequest(BaseModel):
    lifecycle_model_id: int
    to_stage: str


class RetrainRequest(BaseModel):
    scenario: str = "random_holdout"


def _build_orchestrator(
    *,
    scenario: str,
    current_csv_path: str | None = None,
    max_high_psi: int = 0,
    max_ks: int = 2,
    max_chi2: int = 3,
) -> Orchestrator:
    cfg = OrchestratorConfig(
        scenario=scenario,
        current_csv_path=current_csv_path,
        max_high_psi_features=max_high_psi,
        max_ks_significant_numeric=max_ks,
        max_chi2_significant_categorical=max_chi2,
    )
    return Orchestrator(cfg)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Trustworthy AI Monitor API",
        version="0.1.0",
        description="Unified API for drift, orchestration, retraining, lifecycle, and data management.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/overview")
    def overview(limit_runs: int = Query(default=10, ge=1, le=200)) -> dict:
        orch = OrchestratorConfig()
        run_store = RunStore(orch.sqlite_path)  # type: ignore[arg-type]
        lifecycle = default_lifecycle_service()
        dm = default_data_management_service()

        runs = run_store.recent(limit=limit_runs)
        models = lifecycle.list_models()
        experiments = lifecycle.list_experiments(limit=20)
        datasets = dm.list_datasets(limit=20)
        baselines = dm.list_baselines(limit=20)
        production_id = lifecycle.get_production_model_id()

        return {
            "kpis": {
                "n_runs": len(runs),
                "n_models": len(models),
                "n_experiments": len(experiments),
                "n_datasets": len(datasets),
                "n_baselines": len(baselines),
                "production_model_row_id": production_id,
                "last_run_policy_triggered": runs[0].policy_triggered if runs else None,
            },
            "last_run": asdict(runs[0]) if runs else None,
            "latest_model": asdict(models[0]) if models else None,
            "latest_experiment": asdict(experiments[0]) if experiments else None,
        }

    @app.get("/api/orchestration/runs")
    def orchestration_runs(limit: int = Query(default=50, ge=1, le=500)) -> dict:
        cfg = OrchestratorConfig()
        store = RunStore(cfg.sqlite_path)  # type: ignore[arg-type]
        rows = [asdict(r) for r in store.recent(limit=limit)]
        return {"items": rows, "count": len(rows)}

    @app.get("/api/ops/stats")
    def ops_stats() -> dict:
        cfg = OrchestratorConfig()
        store = RunStore(cfg.sqlite_path)  # type: ignore[arg-type]
        stats = store.aggregate_stats()
        total = stats["total_runs"]
        trigger_rate = (stats["triggered_runs"] / total) if total > 0 else 0.0
        return {
            "orchestration": {
                **stats,
                "trigger_rate": round(trigger_rate, 4),
            }
        }

    @app.post("/api/orchestration/check-once")
    def orchestration_check_once(
        scenario: str = Query(default="random_holdout"),
        current_csv_path: str | None = Query(default=None),
        max_high_psi: int = Query(default=0, ge=0),
        max_ks: int = Query(default=2, ge=0),
        max_chi2: int = Query(default=3, ge=0),
    ) -> dict:
        orch = _build_orchestrator(
            scenario=scenario,
            current_csv_path=current_csv_path,
            max_high_psi=max_high_psi,
            max_ks=max_ks,
            max_chi2=max_chi2,
        )
        try:
            result = orch.run_pipeline()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return asdict(result)

    @app.post("/api/retraining/run")
    def retraining_run(req: RetrainRequest) -> dict:
        if req.scenario not in {"random_holdout", "age_shift"}:
            raise HTTPException(status_code=400, detail="scenario must be random_holdout or age_shift")
        ref, cur = split_labeled_reference_current(
            test_size=0.3,
            random_state=42,
            scenario=req.scenario,
        )
        result = run_retrain_pipeline(ref, cur, scenario=req.scenario)
        return {
            "version": result.version,
            "promoted": result.promoted,
            "promote_reason": result.promote_reason,
            "metrics": result.metrics,
            "artifact_path": str(result.artifact_path),
        }

    @app.get("/api/lifecycle/models")
    def lifecycle_models(stage: str | None = None) -> dict:
        svc = default_lifecycle_service()
        stage_enum = DeploymentStage(stage) if stage else None
        items = []
        for m in svc.list_models(stage=stage_enum):
            d = asdict(m)
            d["stage"] = m.stage.value
            items.append(d)
        return {"items": items, "count": len(items)}

    @app.get("/api/lifecycle/experiments")
    def lifecycle_experiments(limit: int = Query(default=50, ge=1, le=500)) -> dict:
        svc = default_lifecycle_service()
        rows = [asdict(e) for e in svc.list_experiments(limit=limit)]
        return {"items": rows, "count": len(rows)}

    @app.get("/api/lifecycle/production")
    def lifecycle_production() -> dict:
        svc = default_lifecycle_service()
        return {"production_model_row_id": svc.get_production_model_id()}

    @app.post("/api/lifecycle/promote")
    def lifecycle_promote(req: PromoteRequest) -> dict:
        svc = default_lifecycle_service()
        try:
            svc.promote_stage(req.lifecycle_model_id, DeploymentStage(req.to_stage))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"ok": True}

    @app.get("/api/data/datasets")
    def data_datasets(limit: int = Query(default=50, ge=1, le=500)) -> dict:
        dm = default_data_management_service()
        rows = [asdict(r) for r in dm.list_datasets(limit=limit)]
        return {"items": rows, "count": len(rows)}

    @app.get("/api/data/baselines")
    def data_baselines(limit: int = Query(default=50, ge=1, le=500)) -> dict:
        dm = default_data_management_service()
        rows = [asdict(r) for r in dm.list_baselines(limit=limit)]
        return {"items": rows, "count": len(rows)}

    @app.get("/api/data/provenance")
    def data_provenance(limit: int = Query(default=50, ge=1, le=500)) -> dict:
        dm = default_data_management_service()
        rows = dm.list_provenance(limit=limit)
        return {"items": rows, "count": len(rows)}

    return app


app = create_app()

