def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_valid_response_shape(client, valid_customer):
    resp = client.post("/predict", json=valid_customer)
    assert resp.status_code == 200

    body = resp.json()
    assert body["prediction"] in ("Churn", "No Churn")
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["threshold_used"] == 0.285
    assert len(body["top_contributing_factors"]) == 5
    for factor in body["top_contributing_factors"]:
        assert "feature" in factor and "shap_value" in factor


def test_predict_high_risk_profile_flags_churn(client, valid_customer):
    """Short tenure + month-to-month + electronic check should predict
    Churn — this is the exact profile verified manually earlier."""
    resp = client.post("/predict", json=valid_customer)
    body = resp.json()
    assert body["prediction"] == "Churn"
    assert body["churn_probability"] > 0.285


def test_predict_low_risk_profile_flags_no_churn(client):
    low_risk = {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "Yes", "tenure": 65, "PhoneService": "Yes",
        "MultipleLines": "Yes", "InternetService": "DSL",
        "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
        "DeviceProtection": "Yes", "TechSupport": "Yes",
        "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Two year", "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 70.0, "TotalCharges": 4550.0,
    }
    resp = client.post("/predict", json=low_risk)
    body = resp.json()
    assert body["prediction"] == "No Churn"
    assert body["churn_probability"] < 0.285


def test_predict_rejects_unseen_category(client, valid_customer):
    bad = dict(valid_customer, PaymentMethod="Cryptocurrency")
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_rejects_missing_field(client, valid_customer):
    bad = dict(valid_customer)
    del bad["TotalCharges"]
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_rate_limit_triggers_after_threshold(client, valid_customer):
    """PREDICT_RATE_LIMIT is 100/minute — send 101 rapid requests and
    confirm the 101st gets throttled (429), not silently accepted."""
    responses = [client.post("/predict", json=valid_customer) for _ in range(101)]
    statuses = [r.status_code for r in responses]
    assert 429 in statuses, "Expected at least one 429 after exceeding the rate limit"