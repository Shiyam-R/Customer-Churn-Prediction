"""
app/model_loader.py
─────────────────────────────────────────────────────────────────────────────
Loads the model + feature column reference exactly once at application
startup and exposes a singleton. Every prediction request reuses these
objects rather than reloading from disk (or worse, retraining) per request.
"""

import json
from dataclasses import dataclass, field
from typing import Any, List

import joblib
import shap

from app.config import MODEL_FILE, FEATURE_COLUMNS_FILE
from app.exceptions import ArtifactLoadError
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelArtifacts:
    model: Any = None
    feature_columns: List[str] = field(default_factory=list)
    explainer: Any = None
    loaded: bool = False


# Module-level singleton — populated by load_artifacts()
artifacts = ModelArtifacts()


def load_artifacts() -> ModelArtifacts:
    """Called once from main.py's lifespan context manager."""
    global artifacts

    logger.info("Loading churn model artifacts...")

    if not MODEL_FILE.exists():
        raise ArtifactLoadError("churn_model.pkl", f"File not found: {MODEL_FILE}")
    try:
        model = joblib.load(MODEL_FILE)
        logger.info("  Model loaded (%s)", MODEL_FILE.name)
    except Exception as exc:
        raise ArtifactLoadError("churn_model.pkl", str(exc)) from exc

    if not FEATURE_COLUMNS_FILE.exists():
        raise ArtifactLoadError("model_columns.json", f"File not found: {FEATURE_COLUMNS_FILE}")
    try:
        with open(FEATURE_COLUMNS_FILE) as f:
            feature_columns = json.load(f)
        logger.info("  Feature columns loaded (%d columns)", len(feature_columns))
    except Exception as exc:
        raise ArtifactLoadError("model_columns.json", str(exc)) from exc

    explainer = shap.TreeExplainer(model)
    logger.info("  SHAP TreeExplainer initialized")

    artifacts.model = model
    artifacts.feature_columns = feature_columns
    artifacts.explainer = explainer
    artifacts.loaded = True

    logger.info("Artifact loading complete.")
    return artifacts
