import pytest
from fastapi.testclient import TestClient

from app.main import app


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
