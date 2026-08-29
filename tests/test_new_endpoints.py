from app.drift_tracker import _live_buffer, record_request
from app.model_loader import artifacts
from app.utils.feature_engineering import prepare_inference_features


def test_root_endpoint_lists_expected_routes(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"]
    assert "/predict" in body["endpoints"]
    assert "/drift" in body["endpoints"]


def test_version_endpoint_reports_model_metadata(client):
    resp = client.get("/version")
    assert resp.status_code == 200
    body = resp.json()

    assert body["api_version"]
    assert body["model_version"] == "1.0.0"
    # trained_at should be a real ISO timestamp, not the "unknown" fallback —
    # confirms model_metadata.json actually loaded, not silently defaulted
    assert body["model_trained_at"] != "unknown"
    assert "T" in body["model_trained_at"]
    # git_sha/environment fall back to "unknown"/"development" locally
    # since no Docker build-arg is set in this test environment
    assert body["git_sha"] == "unknown"
    assert body["environment"] == "development"


def test_drift_reports_insufficient_data_before_threshold(client):
    """Fresh buffer (no /predict calls yet in this test) should report
    insufficient_data, not attempt a meaningless comparison."""
    _live_buffer.clear()
    resp = client.get("/drift")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "insufficient_data"
    assert body["live_sample_size"] == 0
    assert body["features"] == []


def test_predict_feeds_the_drift_buffer(client, valid_customer):
    """Confirms the actual wiring in prediction_service.py — a few real
    /predict calls (well under the 30/minute rate limit) should grow the
    live buffer by exactly that many entries."""
    _live_buffer.clear()
    for _ in range(3):
        resp = client.post("/predict", json=valid_customer)
        assert resp.status_code == 200
    assert len(_live_buffer) == 3


def test_drift_reports_ok_after_enough_live_requests(client, valid_customer):
    """Populates the buffer directly via record_request() — bypassing
    /predict's rate limiter entirely, since that's a separate concern
    from whether drift computation itself works with enough data. Uses
    the same prepare_inference_features the API actually calls, so the
    feature vector shape matches what /drift will really see."""
    _live_buffer.clear()
    X = prepare_inference_features(valid_customer, artifacts.feature_columns)
    for _ in range(35):
        record_request(X.iloc[0])

    resp = client.get("/drift")
    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] == "ok"
    assert body["live_sample_size"] >= 30
    assert len(body["features"]) > 0
    for feat in body["features"]:
        assert feat["severity"] in ("none", "moderate", "significant")
        assert feat["psi"] >= 0
