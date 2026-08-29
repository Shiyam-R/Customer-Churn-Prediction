"""
app/model_loader.py
─────────────────────────────────────────────────────────────────────────────
Loads the model + feature column reference exactly once at application
startup and exposes a singleton. Every prediction request reuses these
objects rather than reloading from disk (or worse, retraining) per request.
"""

import json
from dataclasses import dataclass, field
from typing import Any

import shap
from xgboost import XGBClassifier

from app.config import (
    BASELINE_STATS_FILE,
    FEATURE_COLUMNS_FILE,
    MODEL_FILE,
    MODEL_METADATA_FILE,
)
from app.exceptions import ArtifactLoadError
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelArtifacts:
    model: Any = None
    feature_columns: list[str] = field(default_factory=list)
    explainer: Any = None
    metadata: dict = field(default_factory=dict)
    baseline_stats: dict = field(default_factory=dict)
    loaded: bool = False


# Module-level singleton — populated by load_artifacts()
artifacts = ModelArtifacts()


def load_artifacts() -> ModelArtifacts:
    """Called once from main.py's lifespan context manager."""
    logger.info("Loading churn model artifacts...")

    if not MODEL_FILE.exists():
        raise ArtifactLoadError("churn_model.json", f"File not found: {MODEL_FILE}")
    try:
        model = XGBClassifier()
        model.load_model(MODEL_FILE)
        logger.info("  Model loaded (%s)", MODEL_FILE.name)
    except Exception as exc:
        raise ArtifactLoadError("churn_model.json", str(exc)) from exc

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

    if not MODEL_METADATA_FILE.exists():
        raise ArtifactLoadError("model_metadata.json", f"File not found: {MODEL_METADATA_FILE}")
    try:
        with open(MODEL_METADATA_FILE) as f:
            metadata = json.load(f)
        logger.info("  Model metadata loaded (trained_at=%s)", metadata.get("trained_at"))
    except Exception as exc:
        raise ArtifactLoadError("model_metadata.json", str(exc)) from exc

    if not BASELINE_STATS_FILE.exists():
        raise ArtifactLoadError("baseline_stats.json", f"File not found: {BASELINE_STATS_FILE}")
    try:
        with open(BASELINE_STATS_FILE) as f:
            baseline_stats = json.load(f)
        logger.info("  Baseline drift stats loaded (%d features)", len(baseline_stats))
    except Exception as exc:
        raise ArtifactLoadError("baseline_stats.json", str(exc)) from exc

    artifacts.model = model
    artifacts.feature_columns = feature_columns
    artifacts.explainer = explainer
    artifacts.metadata = metadata
    artifacts.baseline_stats = baseline_stats
    artifacts.loaded = True

    logger.info("Artifact loading complete.")
    return artifacts
