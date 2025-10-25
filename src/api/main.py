"""
FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path

from src.api.routes import (
    prediction_router,
    monitoring_router,
    management_router,
    set_model
)
from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Trustworthy AI Monitor API",
    description="MLOps API for model monitoring, reliability, and fairness",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(prediction_router)
app.include_router(monitoring_router)
app.include_router(management_router)


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("Starting Trustworthy AI Monitor API...")
    logger.info(f"Environment: {config.environment}")
    
    # Load model if exists
    try:
        import joblib
        model_path = Path("models/trained_model.pkl")
        
        if model_path.exists():
            model = joblib.load(model_path)
            set_model(model)
            logger.info("Model loaded successfully")
        else:
            logger.warning("No trained model found. Use /manage/load endpoint to load a model.")
    except Exception as e:
        logger.warning(f"Could not load model on startup: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Trustworthy AI Monitor API...")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Trustworthy AI Monitor API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/manage/health"
    }


@app.get("/status")
async def status():
    """Status endpoint."""
    return {
        "status": "running",
        "api_version": "0.1.0",
        "environment": config.environment
    }


def main():
    """Run the API server."""
    uvicorn.run(
        "src.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload,
        log_level=config.api.log_level
    )


if __name__ == "__main__":
    main()

