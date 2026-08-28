import pytest
from pydantic import ValidationError

from app.schemas.request import CustomerRecord

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


def test_valid_record_passes():
    record = CustomerRecord(**VALID)
    assert record.PaymentMethod == "Electronic check"


def test_unseen_payment_method_rejected():
    """The exact bug found and fixed earlier: an unseen category must be
    rejected by Pydantic, not silently dropped during one-hot alignment."""
    bad = dict(VALID, PaymentMethod="Cryptocurrency")
    with pytest.raises(ValidationError):
        CustomerRecord(**bad)


def test_invalid_contract_rejected():
    bad = dict(VALID, Contract="Weekly")
    with pytest.raises(ValidationError):
        CustomerRecord(**bad)


def test_missing_required_field_rejected():
    bad = dict(VALID)
    del bad["TotalCharges"]
    with pytest.raises(ValidationError):
        CustomerRecord(**bad)


def test_senior_citizen_restricted_to_0_or_1():
    bad = dict(VALID, SeniorCitizen=2)
    with pytest.raises(ValidationError):
        CustomerRecord(**bad)
