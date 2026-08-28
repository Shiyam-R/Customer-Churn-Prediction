"""
app/main.py
─────────────────────────────────────────────────────────────────────────────
FastAPI application factory.

Start the server:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Interactive docs:
    http://localhost:8000/docs      — Swagger UI
    http://localhost:8000/redoc     — ReDoc
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import API_DESCRIPTION, API_TITLE, API_VERSION
from app.exceptions import ChurnAPIError
from app.model_loader import load_artifacts
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Load model artifacts exactly once before the API accepts requests.
    A failure here is loud (logged + raised), not a silent import-time crash.
    """
    logger.info("=" * 60)
    logger.info("Customer Churn Prediction API — starting up...")
    logger.info("=" * 60)

    try:
        load_artifacts()
        logger.info("Startup complete. API is ready to serve requests.")
    except Exception as exc:
        logger.critical("Startup failed: %s", exc, exc_info=True)
        raise

    yield  # API is live here

    logger.info("Customer Churn Prediction API — shutting down.")


# ── Application Factory ───────────────────────────────────────────────────────
def create_app() -> FastAPI:
    application = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Centralised exception handler ─────────────────────────────────────────
    @application.exception_handler(ChurnAPIError)
    async def churn_error_handler(request: Request, exc: ChurnAPIError) -> JSONResponse:
        """Convert any ChurnAPIError subclass into a structured JSON response."""
        logger.warning("Handled error [%s] — %s", exc.status_code, exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "status": "error",
                "message": exc.message,
                "detail": exc.detail,
                "code": exc.status_code,
            },
        )

    application.include_router(router)
    return application


app = create_app()
