"""FastAPI app that exposes orchestration, retraining, lifecycle, and data APIs."""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.data_management.service import default_data_management_service
from src.drift_detection.schema import ADULT_CATEGORICAL_FEATURES, ADULT_NUMERIC_FEATURES
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


class PredictRequest(BaseModel):
    rows: list[dict[str, Any]]


def _expected_feature_columns() -> list[str]:
    return list(ADULT_NUMERIC_FEATURES) + list(ADULT_CATEGORICAL_FEATURES)


@lru_cache(maxsize=16)
def _load_model_cached(artifact_path: str, mtime_ns: int):
    # mtime_ns is part of the cache key so model file updates invalidate cache.
    del mtime_ns
    return joblib.load(artifact_path)


def _load_model_from_artifact(artifact_path: Path):
    return _load_model_cached(str(artifact_path), artifact_path.stat().st_mtime_ns)


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

    @app.post("/api/inference/predict")
    def inference_predict(req: PredictRequest) -> dict:
        if not req.rows:
            raise HTTPException(status_code=400, detail="rows must contain at least one record")
        expected = _expected_feature_columns()
        X = pd.DataFrame(req.rows)
        missing = [c for c in expected if c not in X.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required feature columns: {', '.join(missing)}",
            )
        X = X[expected]

        lifecycle = default_lifecycle_service()
        production_id = lifecycle.get_production_model_id()
        if production_id is None:
            raise HTTPException(
                status_code=409,
                detail="No production model is set. Promote a model to production first.",
            )
        model_row = lifecycle.get_model_by_id(production_id)
        if model_row is None:
            raise HTTPException(status_code=404, detail=f"Production model row {production_id} not found")

        artifact_path = Path(model_row.artifact_path)
        if not artifact_path.is_file():
            raise HTTPException(
                status_code=500,
                detail=f"Production artifact not found at {artifact_path.as_posix()}",
            )
        model = _load_model_from_artifact(artifact_path)
        try:
            preds = model.predict(X)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e

        probs: list[float] | None = None
        if hasattr(model, "predict_proba"):
            try:
                arr = model.predict_proba(X)
                probs = [float(row[-1]) for row in arr]
            except Exception:
                probs = None

        pred_values = preds.tolist() if hasattr(preds, "tolist") else list(preds)
        pred_labels = [int(p) if str(p).isdigit() else p for p in pred_values]
        pred_income = [
            ">50K" if str(p) in {"1", "1.0"} or p == 1 else "<=50K"
            for p in pred_labels
        ]

        return {
            "model_row_id": model_row.id,
            "model_version_num": model_row.version_num,
            "n_rows": len(X),
            "predictions": pred_labels,
            "predicted_income_class": pred_income,
            "positive_class_probability": probs,
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

