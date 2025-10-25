"""
API routes for model serving and monitoring.
"""

from fastapi import APIRouter, HTTPException, Depends
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.api.schemas import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    PerformanceMetrics, DriftMetrics, FairnessMetrics,
    MonitoringReport, HealthResponse, ModelInfo
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Router instances
prediction_router = APIRouter(prefix="/predict", tags=["Predictions"])
monitoring_router = APIRouter(prefix="/monitor", tags=["Monitoring"])
management_router = APIRouter(prefix="/manage", tags=["Management"])


# Global model and preprocessor (to be set by main app)
_model = None
_preprocessor = None
_feature_names = []


def set_model(model: Any, preprocessor: Any = None, feature_names: List[str] = None):
    """Set global model and preprocessor."""
    global _model, _preprocessor, _feature_names
    _model = model
    _preprocessor = preprocessor
    _feature_names = feature_names or []
    logger.info("Model and preprocessor set in routes")


def get_model():
    """Dependency to get model."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _model


def get_preprocessor():
    """Dependency to get preprocessor."""
    return _preprocessor


@prediction_router.post("/single", response_model=PredictionResponse)
async def predict_single(
    request: PredictionRequest,
    model = Depends(get_model)
) -> PredictionResponse:
    """
    Make a single prediction.
    """
    try:
        # Convert features to DataFrame
        features_df = pd.DataFrame([request.features])
        
        # Preprocess if preprocessor is available
        if _preprocessor is not None:
            features_df = _preprocessor.transform(features_df)
        
        # Make prediction
        prediction = int(model.predict(features_df)[0])
        
        # Get probability if available
        probability = None
        if hasattr(model, 'predict_proba'):
            probability = float(model.predict_proba(features_df)[0, 1])
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@prediction_router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    model = Depends(get_model)
) -> BatchPredictionResponse:
    """
    Make batch predictions.
    """
    try:
        # Convert features to DataFrame
        features_df = pd.DataFrame(request.features)
        
        # Preprocess if preprocessor is available
        if _preprocessor is not None:
            features_df = _preprocessor.transform(features_df)
        
        # Make predictions
        predictions = model.predict(features_df).tolist()
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_df)[:, 1].tolist()
        
        return BatchPredictionResponse(
            predictions=predictions,
            probabilities=probabilities,
            count=len(predictions),
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@monitoring_router.get("/performance", response_model=Dict[str, Any])
async def get_performance_metrics(
    model = Depends(get_model)
) -> Dict[str, Any]:
    """
    Get current performance metrics.
    Note: Requires ground truth data to be provided separately.
    """
    # This is a placeholder - in production, you'd fetch stored metrics
    return {
        "message": "Performance metrics endpoint",
        "note": "Requires ground truth data for evaluation",
        "timestamp": datetime.now().isoformat()
    }


@monitoring_router.get("/drift", response_model=Dict[str, Any])
async def get_drift_metrics() -> Dict[str, Any]:
    """
    Get current drift metrics.
    """
    # This is a placeholder - in production, you'd calculate drift
    return {
        "message": "Drift metrics endpoint",
        "note": "Requires baseline and current data for drift detection",
        "timestamp": datetime.now().isoformat()
    }


@monitoring_router.get("/fairness", response_model=Dict[str, Any])
async def get_fairness_metrics() -> Dict[str, Any]:
    """
    Get current fairness metrics.
    """
    # This is a placeholder - in production, you'd calculate fairness
    return {
        "message": "Fairness metrics endpoint",
        "note": "Requires sensitive features and predictions for fairness analysis",
        "timestamp": datetime.now().isoformat()
    }


@monitoring_router.get("/report", response_model=Dict[str, Any])
async def get_monitoring_report() -> Dict[str, Any]:
    """
    Get comprehensive monitoring report.
    """
    return {
        "message": "Comprehensive monitoring report",
        "performance": await get_performance_metrics(_model),
        "drift": await get_drift_metrics(),
        "fairness": await get_fairness_metrics(),
        "timestamp": datetime.now().isoformat()
    }


@management_router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    """
    return HealthResponse(
        status="healthy" if _model is not None else "model_not_loaded",
        model_loaded=_model is not None,
        version="0.1.0",
        timestamp=datetime.now().isoformat()
    )


@management_router.get("/info", response_model=Dict[str, Any])
async def get_model_info(model = Depends(get_model)) -> Dict[str, Any]:
    """
    Get model information.
    """
    return {
        "model_type": type(model).__name__,
        "features": _feature_names,
        "feature_count": len(_feature_names),
        "version": "0.1.0",
        "timestamp": datetime.now().isoformat()
    }


__all__ = [
    "prediction_router",
    "monitoring_router",
    "management_router",
    "set_model"
]

