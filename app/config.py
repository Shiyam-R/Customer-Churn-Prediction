"""
app/config.py
─────────────────────────────────────────────────────────────────────────────
Central place for paths, the decision threshold, and API metadata.
Nothing elsewhere in the app should hardcode these.
"""

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

# Cost-sensitive threshold — FN cost (missed churner) ~7x FP cost (unneeded
# email), chosen via the sweep in hyperparameter_tuning.py. Not 0.5.
THRESHOLD = 0.285

# Rate limit for /predict — each request runs a real SHAP TreeExplainer
# computation, not just a cheap predict_proba() call. This is a security/
# cost consideration, not a UX one: unrestricted hammering of this endpoint
# is a real resource-exhaustion vector given the per-request compute cost.
PREDICT_RATE_LIMIT = "30/minute"