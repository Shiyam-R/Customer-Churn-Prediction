"""
app/exceptions.py
─────────────────────────────────────────────────────────────────────────────
Custom exception hierarchy. Every exception carries its own HTTP status
code so main.py's exception handler can return a consistent structured
JSON error without try/except boilerplate in the route itself.
"""



class ChurnAPIError(Exception):
    """Base class for all Churn API errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.detail = detail or message


# ── Artifact / Model Errors (503) ────────────────────────────────────────────
class ArtifactLoadError(ChurnAPIError):
    """Raised when a required artifact fails to load at startup."""

    def __init__(self, artifact: str, reason: str) -> None:
        super().__init__(
            message=f"Failed to load artifact: {artifact}.",
            status_code=503,
            detail=f"Artifact '{artifact}' could not be loaded. Reason: {reason}.",
        )


class ModelNotLoadedError(ChurnAPIError):
    """Raised when a prediction is requested before/without a loaded model."""

    def __init__(self) -> None:
        super().__init__(
            message="Model is not loaded.",
            status_code=503,
            detail="The churn model artifact was not found or failed to load during startup.",
        )


# ── Inference Errors (500) ────────────────────────────────────────────────────
class PreprocessingError(ChurnAPIError):
    """Raised when feature engineering fails on an otherwise-valid request."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            message="Preprocessing pipeline failed.",
            status_code=500,
            detail=f"Feature engineering error: {reason}.",
        )


class PredictionError(ChurnAPIError):
    """Raised when model inference itself fails."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            message="Prediction failed.",
            status_code=500,
            detail=f"Model inference error: {reason}.",
        )
