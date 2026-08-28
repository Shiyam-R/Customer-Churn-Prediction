"""
app/services/prediction_service.py
─────────────────────────────────────────────────────────────────────────────
The actual prediction logic — preprocessing, inference, SHAP explanation.
Kept separate from app/api/routes.py so the route stays a thin HTTP layer
and this logic is testable/reusable without spinning up FastAPI at all.
"""

from app.config import THRESHOLD
from app.exceptions import ModelNotLoadedError, PreprocessingError, PredictionError
from app.model_loader import artifacts
from app.schemas.request import CustomerRecord
from app.schemas.response import ContributingFactor, PredictionResponse
from app.utils.feature_engineering import prepare_inference_features
from app.utils.logger import get_logger

logger = get_logger(__name__)


def predict_churn(record: CustomerRecord) -> PredictionResponse:
    if not artifacts.loaded:
        raise ModelNotLoadedError()

    raw = record.model_dump()

    try:
        X = prepare_inference_features(raw, artifacts.feature_columns)
    except Exception as exc:
        logger.error("Preprocessing failed: %s", exc, exc_info=True)
        raise PreprocessingError(str(exc)) from exc

    try:
        proba = float(artifacts.model.predict_proba(X)[:, 1][0])
        shap_values = artifacts.explainer(X)
    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise PredictionError(str(exc)) from exc

    prediction = "Churn" if proba >= THRESHOLD else "No Churn"
    contributions = sorted(
        zip(X.columns, shap_values.values[0]),
        key=lambda x: -abs(x[1])
    )[:5]

    logger.info("Prediction: %s (proba=%.4f)", prediction, proba)

    return PredictionResponse(
        prediction=prediction,
        churn_probability=round(proba, 4),
        threshold_used=THRESHOLD,
        top_contributing_factors=[
            ContributingFactor(feature=feat, shap_value=round(float(val), 4))
            for feat, val in contributions
        ],
    )
