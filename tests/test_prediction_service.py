import pytest

from app.exceptions import ModelNotLoadedError
from app.model_loader import artifacts
from app.schemas.request import CustomerRecord
from app.services.prediction_service import predict_churn

VALID = {
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


def test_predict_churn_raises_when_model_not_loaded(monkeypatch):
    """Simulates a request arriving before/without successful startup —
    should fail loudly (ModelNotLoadedError -> 503), not silently."""
    monkeypatch.setattr(artifacts, "loaded", False)
    record = CustomerRecord(**VALID)
    with pytest.raises(ModelNotLoadedError):
        predict_churn(record)
