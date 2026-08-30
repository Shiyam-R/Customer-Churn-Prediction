"""
app/config.py
─────────────────────────────────────────────────────────────────────────────
Central place for paths, the decision threshold, and API metadata.
Nothing elsewhere in the app should hardcode these.
"""

import os
from pathlib import Path

API_TITLE = "Customer Churn Prediction API"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "Predicts customer churn probability from account/service attributes, "
    "with a per-request SHAP explanation of the top contributing factors."
)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_FILE = ARTIFACTS_DIR / "churn_model.json"
FEATURE_COLUMNS_FILE = ARTIFACTS_DIR / "model_columns.json"
MODEL_METADATA_FILE = ARTIFACTS_DIR / "model_metadata.json"
BASELINE_STATS_FILE = ARTIFACTS_DIR / "baseline_stats.json"

# Cost-sensitive threshold — FN cost (missed churner) ~7x FP cost (unneeded
# email), chosen via the sweep in hyperparameter_tuning.py. Not 0.5.
THRESHOLD = 0.345

# Rate limit for /predict — each request runs a real SHAP TreeExplainer
# computation, not just a cheap predict_proba() call. This is a security/
# cost consideration, not a UX one: unrestricted hammering of this endpoint
# is a real resource-exhaustion vector given the per-request compute cost.
# Configurable via env var — measured latency is only ~3-23ms/request even
# under concurrent load (see load_tests/LOAD_TEST_RESULTS.md), so 100/min
# is still conservative, not a bottleneck. Override for load testing with
# a different concurrency profile, e.g. PREDICT_RATE_LIMIT=1000/minute.
PREDICT_RATE_LIMIT = os.getenv("PREDICT_RATE_LIMIT", "100/minute")

# /version identity — GIT_SHA is injected via a Docker build-arg (see
# Dockerfile + ci.yml's publish-and-deploy job); falls back to "unknown"
# for local runs where it's never set. ENVIRONMENT defaults to
# "production" inside the Docker image (set via ENV) and "development"
# for a bare local `uvicorn --reload` run that never sets it.
GIT_SHA = os.getenv("GIT_SHA", "unknown")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# /drift — minimum live requests buffered before a report is meaningful,
# and how many recent requests to retain.
DRIFT_MIN_SAMPLES = 30
DRIFT_BUFFER_SIZE = 500