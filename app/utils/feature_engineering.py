"""
app/utils/feature_engineering.py
─────────────────────────────────────────────────────────────────────────────
Inference-only version of the training pipeline's feature engineering.
Deliberately does NOT import prepare_data.py or scikit-learn's split
utilities — the serving app has no business depending on training-only
code. Encoding logic here must stay in lockstep with the training-side
feature_engineering.py; any change there needs to be mirrored here.
"""

import pandas as pd

REDUNDANT_TERNARY_COLS = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
SIMPLE_BINARY_COLS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
ORDINAL_MAP = {"Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2}}
NOMINAL_COLS = ["InternetService", "PaymentMethod"]


def consolidate_and_binarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in REDUNDANT_TERNARY_COLS:
        df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})
    df["gender"] = df["gender"].map({"Female": 1, "Male": 0})
    for col in REDUNDANT_TERNARY_COLS + [c for c in SIMPLE_BINARY_COLS if c != "gender"]:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                     "TechSupport", "StreamingTV", "StreamingMovies"]
    df["num_services"] = df[service_cols].sum(axis=1)
    df["has_family"] = ((df["Partner"] == 1) | (df["Dependents"] == 1)).astype(int)
    df["high_risk_combo"] = (
        (df["Contract"] == "Month-to-month") & (df["PaymentMethod"] == "Electronic check")
    ).astype(int)
    return df


def ordinal_encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, mapping in ORDINAL_MAP.items():
        df[col] = df[col].map(mapping)
    return df


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, columns=NOMINAL_COLS, drop_first=True)


def prepare_inference_features(raw_record: dict, reference_columns: list) -> pd.DataFrame:
    """
    Applies the exact training-time pipeline to a single raw customer
    record. reference_columns (loaded from model_columns.json at startup)
    is what makes this leakage/skew-safe: any dummy column missing from
    this single row gets filled with 0, and any unexpected column (there
    shouldn't be any now that Pydantic Literal types validate categories
    up front) gets dropped rather than silently misread by the model.
    """
    df = pd.DataFrame([raw_record])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["tenure"] == 0, "TotalCharges"] = 0
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df = consolidate_and_binarize(df)
    df = create_features(df)
    df = ordinal_encode(df)
    df = one_hot_encode(df)

    df = df.reindex(columns=reference_columns, fill_value=0)
    return df
