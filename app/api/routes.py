"""
app/api/routes.py
─────────────────────────────────────────────────────────────────────────────
Thin route handlers. All actual logic lives in app/services/ — routes
here just wire HTTP verbs/paths to that logic.
"""

from fastapi import APIRouter

from app.model_loader import artifacts
from app.schemas.request import CustomerRecord
from app.schemas.response import HealthResponse, PredictionResponse
from app.services.prediction_service import predict_churn

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(record: CustomerRecord) -> PredictionResponse:
    return predict_churn(record)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=artifacts.loaded)
