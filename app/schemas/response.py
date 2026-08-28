"""
app/schemas/response.py
─────────────────────────────────────────────────────────────────────────────
Response contracts. Using explicit models (not bare dicts) means FastAPI
validates and documents the output shape too, not just the input.
"""

from typing import List
from pydantic import BaseModel


class ContributingFactor(BaseModel):
    feature: str
    shap_value: float


class PredictionResponse(BaseModel):
    prediction: str
    churn_probability: float
    threshold_used: float
    top_contributing_factors: List[ContributingFactor]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
