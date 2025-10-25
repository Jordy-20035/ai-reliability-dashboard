"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request schema for predictions."""
    features: Dict[str, Any] = Field(..., description="Feature values for prediction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "age": 35,
                    "education": "Bachelors",
                    "occupation": "Tech-support"
                }
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    prediction: int = Field(..., description="Predicted class")
    probability: Optional[float] = Field(None, description="Prediction probability")
    timestamp: str = Field(..., description="Prediction timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.75,
                "timestamp": "2025-01-01T12:00:00"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""
    features: List[Dict[str, Any]] = Field(..., description="List of feature dictionaries")
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [
                    {"age": 35, "education": "Bachelors"},
                    {"age": 42, "education": "Masters"}
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: List[int] = Field(..., description="List of predictions")
    probabilities: Optional[List[float]] = Field(None, description="List of probabilities")
    count: int = Field(..., description="Number of predictions")
    timestamp: str = Field(..., description="Prediction timestamp")


class PerformanceMetrics(BaseModel):
    """Performance metrics schema."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float] = None
    n_samples: int
    timestamp: str


class DriftMetrics(BaseModel):
    """Drift metrics schema."""
    drift_detected: bool
    drift_score: float
    features_with_drift: List[str]
    timestamp: str


class FairnessMetrics(BaseModel):
    """Fairness metrics schema."""
    fairness_violations_detected: bool
    total_violations: int
    demographic_parity_difference: Optional[float] = None
    equal_opportunity_difference: Optional[float] = None
    timestamp: str


class MonitoringReport(BaseModel):
    """Comprehensive monitoring report."""
    performance: PerformanceMetrics
    drift: Optional[DriftMetrics] = None
    fairness: Optional[FairnessMetrics] = None
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Response timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "version": "0.1.0",
                "timestamp": "2025-01-01T12:00:00"
            }
        }


class ModelInfo(BaseModel):
    """Model information."""
    model_type: str
    features: List[str]
    trained_at: Optional[str] = None
    version: str


__all__ = [
    "PredictionRequest",
    "PredictionResponse",
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "PerformanceMetrics",
    "DriftMetrics",
    "FairnessMetrics",
    "MonitoringReport",
    "HealthResponse",
    "ModelInfo"
]

