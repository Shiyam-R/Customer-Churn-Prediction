"""
app/api/routes.py
─────────────────────────────────────────────────────────────────────────────
Thin route handlers. All actual logic lives in app/services/ — routes
here just wire HTTP verbs/paths to that logic.
"""

from fastapi import APIRouter, Request

from app.config import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    ENVIRONMENT,
    GIT_SHA,
    PREDICT_RATE_LIMIT,
)
from app.drift_tracker import compute_drift_report
from app.model_loader import artifacts
from app.rate_limiter import limiter
from app.schemas.request import CustomerRecord
from app.schemas.response import (
    DriftResponse,
    HealthResponse,
    PredictionResponse,
    RootResponse,
    VersionResponse,
)
from app.services.prediction_service import predict_churn

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
@limiter.limit(PREDICT_RATE_LIMIT)
def predict(request: Request, record: CustomerRecord) -> PredictionResponse:
    return predict_churn(record)


@router.get("/health", response_model=HealthResponse, tags=["Info"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=artifacts.loaded)


@router.get("/", response_model=RootResponse, tags=["Info"])
def root() -> RootResponse:
    return RootResponse(
        name=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        endpoints=["/", "/health", "/version", "/drift", "/predict", "/docs", "/redoc"],
    )


@router.get("/version", response_model=VersionResponse, tags=["Info"])
def version() -> VersionResponse:
    return VersionResponse(
        api_version=API_VERSION,
        model_version=artifacts.metadata.get("model_version", "unknown"),
        model_trained_at=artifacts.metadata.get("trained_at", "unknown"),
        git_sha=GIT_SHA,
        environment=ENVIRONMENT,
    )


@router.get(
    "/drift",
    response_model=DriftResponse,
    tags=["Monitoring"],
    description=(
        "Reports feature-distribution drift (PSI) between recent live "
        "/predict requests and the training data. SCOPE CAVEAT: live "
        "requests are tracked in an in-process buffer — under a "
        "multi-worker deployment, this only reflects whichever worker "
        "answers this specific request, not a combined view across all "
        "workers. Currently deployed as a single worker, so dormant."
    ),
)
def drift() -> DriftResponse:
    report = compute_drift_report(artifacts.baseline_stats)
    return DriftResponse(**report)