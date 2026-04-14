"""
FraudShield Detection API - FastAPI Application.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import logging

from app.config import API_TITLE, API_DESCRIPTION, API_VERSION, MODEL_VERSION, METRICS_ENABLED
from app.model import FraudDetectionModel
from app.schemas import (
    TransactionRequest,
    FraudPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from app.middleware import MetricsMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model: FraudDetectionModel = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = FraudDetectionModel()
        logger.info("Model loaded successfully at startup")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    yield
    model = None


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if METRICS_ENABLED:
    app.add_middleware(MetricsMiddleware)


@app.get("/", tags=["Info"])
async def root():
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model and model.is_loaded() else "unhealthy",
        model_loaded=model is not None and model.is_loaded(),
        model_version=MODEL_VERSION,
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    if not METRICS_ENABLED:
        raise HTTPException(status_code=503, detail="Metrics disabled")
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=FraudPredictionResponse, tags=["Prediction"])
async def predict(request: TransactionRequest):
    """
    Predict whether a financial transaction is fraudulent.

    Returns fraud probability, risk level (LOW/MEDIUM/HIGH/CRITICAL),
    and prediction latency.
    """
    if model is None or not model.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        is_fraud, probability, risk_level, latency_ms = model.predict(request.model_dump())
        return FraudPredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=probability,
            risk_level=risk_level,
            model_version=MODEL_VERSION,
            latency_ms=latency_ms,
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Info"])
async def model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    info = model.get_info()
    return ModelInfoResponse(**info)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)