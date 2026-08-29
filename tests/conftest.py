import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rate_limiter import limiter


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """slowapi's Limiter storage is a module-level singleton — without
    this, one test's requests count against every later test's budget
    within the same 1-minute window (a real bug found while writing the
    drift tests: an unrelated 3-request test started failing with 429
    because an earlier test had already exhausted the shared 30/minute
    counter). Reset before every test for proper isolation."""
    limiter.limiter.storage.reset()
    yield


@pytest.fixture
def client():
    """TestClient as a context manager — triggers the lifespan startup
    (model loading) exactly like a real server would."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def valid_customer():
    return {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0, "TotalCharges": 170.0,
    }
