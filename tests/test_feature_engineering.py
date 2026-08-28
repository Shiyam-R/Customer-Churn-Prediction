import pandas as pd

from app.utils.feature_engineering import (
    consolidate_and_binarize, create_features, ordinal_encode,
    one_hot_encode, prepare_inference_features,
)


def test_consolidate_and_binarize_maps_yes_no_and_collapses_ternary():
    df = pd.DataFrame([{
        "gender": "Female", "Partner": "Yes", "Dependents": "No",
        "PhoneService": "Yes", "PaperlessBilling": "No",
        "MultipleLines": "No phone service",
        "OnlineSecurity": "No internet service",
        "OnlineBackup": "Yes", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
    }])
    result = consolidate_and_binarize(df)

    assert result["gender"].iloc[0] == 1
    assert result["Partner"].iloc[0] == 1
    assert result["Dependents"].iloc[0] == 0
    # "No phone/internet service" must collapse to plain "No" (0) —
    # this is the redundant-ternary-column consolidation from block 5
    assert result["MultipleLines"].iloc[0] == 0
    assert result["OnlineSecurity"].iloc[0] == 0
    assert result["OnlineBackup"].iloc[0] == 1


def test_create_features_flags_high_risk_combo():
    """Contract=Month-to-month + PaymentMethod=Electronic check is the
    interaction feature found via the EDA heatmap — 0.54 churn rate cell."""
    df = pd.DataFrame([{
        "Contract": "Month-to-month", "PaymentMethod": "Electronic check",
        "Partner": 0, "Dependents": 0,
        "OnlineSecurity": 0, "OnlineBackup": 0, "DeviceProtection": 0,
        "TechSupport": 0, "StreamingTV": 0, "StreamingMovies": 0,
    }])
    result = create_features(df)

    assert result["high_risk_combo"].iloc[0] == 1
    assert result["has_family"].iloc[0] == 0
    assert result["num_services"].iloc[0] == 0


def test_create_features_no_high_risk_combo_for_different_contract():
    df = pd.DataFrame([{
        "Contract": "Two year", "PaymentMethod": "Electronic check",
        "Partner": 1, "Dependents": 0,
        "OnlineSecurity": 1, "OnlineBackup": 1, "DeviceProtection": 0,
        "TechSupport": 0, "StreamingTV": 0, "StreamingMovies": 0,
    }])
    result = create_features(df)

    assert result["high_risk_combo"].iloc[0] == 0
    assert result["has_family"].iloc[0] == 1
    assert result["num_services"].iloc[0] == 2


def test_ordinal_encode_preserves_contract_order():
    df = pd.DataFrame([
        {"Contract": "Two year"},
        {"Contract": "Month-to-month"},
        {"Contract": "One year"},
    ])
    result = ordinal_encode(df)
    assert result["Contract"].tolist() == [2, 0, 1]


def test_one_hot_encode_on_single_row_drops_the_only_category_present():
    """On a single-row frame, drop_first removes whatever category IS
    present — here DSL disappears entirely, since it's the only category
    that exists in this one row. This is exactly why
    prepare_inference_features must reindex against the full training-time
    column list (see the alignment test below): one_hot_encode() alone is
    not safe to call on a single inference row in isolation."""
    df = pd.DataFrame([{"InternetService": "DSL", "PaymentMethod": "Mailed check"}])
    result = one_hot_encode(df)

    assert "InternetService_DSL" not in result.columns
    # Fiber optic/No are absent too — never present in this row, not
    # "dropped" the way DSL was. Confirms zero InternetService_* columns.
    assert not any(c.startswith("InternetService_") for c in result.columns)


def test_one_hot_encode_with_multiple_categories_present():
    """With more than one category actually present (as in real training
    data), drop_first behaves as expected — one category becomes the
    baseline, the others get their own dummy column."""
    df = pd.DataFrame([
        {"InternetService": "DSL", "PaymentMethod": "Mailed check"},
        {"InternetService": "Fiber optic", "PaymentMethod": "Mailed check"},
        {"InternetService": "No", "PaymentMethod": "Mailed check"},
    ])
    result = one_hot_encode(df)

    assert "InternetService_DSL" not in result.columns  # dropped baseline
    assert "InternetService_Fiber optic" in result.columns
    assert "InternetService_No" in result.columns


def test_prepare_inference_features_tenure_zero_guard():
    """tenure=0 customers get TotalCharges=0, not imputed — matches the
    training-time handling (they genuinely haven't been billed yet)."""
    raw = {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 0, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month",
        "PaperlessBilling": "No", "PaymentMethod": "Mailed check",
        "MonthlyCharges": 50.0, "TotalCharges": "",
    }
    result = prepare_inference_features(raw, ["tenure", "TotalCharges", "MonthlyCharges"])
    assert result["TotalCharges"].iloc[0] == 0


def test_prepare_inference_features_aligns_to_reference_columns():
    """The core training-serving skew guardrail: a single request's one-hot
    encoding only produces columns for categories present in THAT row —
    reindexing against reference_columns must fill every missing dummy
    column with 0, not drop or misalign them."""
    raw = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 10, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Two year",
        "PaperlessBilling": "No", "PaymentMethod": "Mailed check",
        "MonthlyCharges": 50.0, "TotalCharges": 500.0,
    }
    reference_columns = ["tenure", "PaymentMethod_Electronic check", "InternetService_Fiber optic"]
    result = prepare_inference_features(raw, reference_columns)

    assert list(result.columns) == reference_columns
    assert result["PaymentMethod_Electronic check"].iloc[0] == 0
    assert result["InternetService_Fiber optic"].iloc[0] == 0
