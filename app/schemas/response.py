"""
app/schemas/response.py
─────────────────────────────────────────────────────────────────────────────
Response contracts. Using explicit models (not bare dicts) means FastAPI
validates and documents the output shape too, not just the input.
"""


from pydantic import BaseModel


class ContributingFactor(BaseModel):
    feature: str
    shap_value: float


class PredictionResponse(BaseModel):
    prediction: str
    churn_probability: float
    threshold_used: float
    top_contributing_factors: list[ContributingFactor]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class RootResponse(BaseModel):
    name: str
    version: str
    description: str
    endpoints: list[str]


class VersionResponse(BaseModel):
    api_version: str
    model_version: str
    model_trained_at: str
    git_sha: str
    environment: str


class DriftFeatureResult(BaseModel):
    feature: str
    psi: float
    severity: str  # "none" | "moderate" | "significant"


class DriftResponse(BaseModel):
    status: str  # "ok" | "insufficient_data"
    live_sample_size: int
    min_samples_required: int
    features: list[DriftFeatureResult]
