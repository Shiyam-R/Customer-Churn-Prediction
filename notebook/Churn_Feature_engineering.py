"""
Phase 1 / Project 1 — Customer Churn: Feature Engineering

Design decisions (from review):
1. Binary Yes/No columns -> map to 1/0
2. 7 columns with a redundant 3rd category ("No internet service" /
   "No phone service") -> consolidated to binary first (that info is
   already captured by InternetService/PhoneService), then mapped to 1/0
3. Contract -> ordinal encode (0/1/2) — genuine order, commitment level
4. InternetService, PaymentMethod -> one-hot, k-1 columns (drop first)
   to avoid the dummy variable trap for logistic regression
5. tenure, MonthlyCharges, TotalCharges -> StandardScaler, FIT ON TRAIN
   ONLY, applied to val/test — same leakage discipline as the data split
6. TotalCharges -> dropped for logistic regression (multicollinear with
   tenure * MonthlyCharges), kept for XGBoost (trees are collinearity-
   robust)

Produces two parallel feature sets: one for logistic regression,
one for XGBoost.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from prepare_data import load_and_clean, stratified_split

# Columns with a redundant 3rd category derivable from InternetService/PhoneService
REDUNDANT_TERNARY_COLS = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

SIMPLE_BINARY_COLS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]

ORDINAL_MAP = {"Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2}}

NOMINAL_COLS = ["InternetService", "PaymentMethod"]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def consolidate_and_binarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Consolidate the redundant ternary columns to binary first
    for col in REDUNDANT_TERNARY_COLS:
        df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})

    # gender needs its own map (Female/Male, not Yes/No)
    df["gender"] = df["gender"].map({"Female": 1, "Male": 0})

    # Everything else in these two groups is now clean Yes/No -> 1/0
    for col in REDUNDANT_TERNARY_COLS + [c for c in SIMPLE_BINARY_COLS if c != "gender"]:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    NEW engineered features (created AFTER consolidate_and_binarize, so the
    6 service columns are already clean 1/0, but BEFORE ordinal/one-hot
    encoding, so Contract/PaymentMethod are still raw strings we can
    condition on directly).
    """
    df = df.copy()

    # 1. num_services — how bundled-in is this customer (0-6 add-on services)
    service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                     "TechSupport", "StreamingTV", "StreamingMovies"]
    df["num_services"] = df[service_cols].sum(axis=1)

    # 2. has_family — household status independent of the two raw columns
    df["has_family"] = ((df["Partner"] == 1) | (df["Dependents"] == 1)).astype(int)

    # 3. high_risk_combo — the specific interaction cell EDA flagged (0.54 churn):
    #    Month-to-month contract + Electronic check payment
    df["high_risk_combo"] = (
        (df["Contract"] == "Month-to-month") & (df["PaymentMethod"] == "Electronic check")
    ).astype(int)

    # NOTE: avg_charge_deviation was tested and rejected. Investigation showed
    # its entire distribution (not just near-zero) is a mechanical proxy for
    # tenure: TotalCharges/tenure is a running average, so its deviation from
    # current MonthlyCharges shrinks as tenure grows (more months diluting any
    # early fluctuation) — same effect as standard error shrinking with sample
    # size. Both the "spike" near 0 and the U-shaped tails traced back to
    # tenure, which is already a direct feature. No independent signal.

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
    Applies the EXACT SAME pipeline used in training to a single raw
    customer record (no target column, no split). Used by the API at
    request time — this is the training-serving skew guardrail.

    raw_record: dict matching the original CSV schema (no customerID, no Churn)
    reference_columns: the exact training-time column list (order matters),
        saved alongside the model artifact. Required because one-hot
        encoding a single row only produces columns for categories present
        in THAT row — reindexing against the saved column list fills in
        every missing dummy column with 0, matching what the model expects.
    """
    df = pd.DataFrame([raw_record])

    # Same TotalCharges handling as load_and_clean, minus the target mapping
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["tenure"] == 0, "TotalCharges"] = 0
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df = consolidate_and_binarize(df)
    df = create_features(df)
    df = ordinal_encode(df)
    df = one_hot_encode(df)

    # Align to training-time columns: adds any missing dummy columns as 0,
    # drops anything unexpected, and fixes column order to match the model.
    df = df.reindex(columns=reference_columns, fill_value=0)

    return df


def scale_numeric(X_train, X_val, X_test, cols=NUMERIC_COLS):
    """Fit scaler on train only. Apply the SAME fitted scaler to val/test."""
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    X_train[cols] = scaler.fit_transform(X_train[cols])
    X_val[cols] = scaler.transform(X_val[cols])
    X_test[cols] = scaler.transform(X_test[cols])

    return X_train, X_val, X_test, scaler


def build_feature_sets():
    """
    Returns two dicts, each with train/val/test X and y:
      - 'logreg': scaled, TotalCharges dropped, k-1 one-hot
      - 'xgboost': unscaled, TotalCharges kept, k-1 one-hot
    (one-hot/ordinal/binary encoding is identical for both — only
    scaling and the TotalCharges decision differ)
    """
    df = load_and_clean()
    df = consolidate_and_binarize(df)
    df = create_features(df)
    df = ordinal_encode(df)
    df = one_hot_encode(df)

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(df)

    # --- XGBoost set: no scaling, keep TotalCharges ---
    xgb_set = {
        "X_train": X_train.copy(), "X_val": X_val.copy(), "X_test": X_test.copy(),
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }

    # --- Logistic regression set: scaled, TotalCharges dropped ---
    lr_X_train = X_train.drop(columns=["TotalCharges"])
    lr_X_val = X_val.drop(columns=["TotalCharges"])
    lr_X_test = X_test.drop(columns=["TotalCharges"])
    # avg_charge_deviation and num_services are continuous/count-scale new
    # features -> need the same scaling treatment as the original numerics
    lr_numeric_cols = [c for c in NUMERIC_COLS if c != "TotalCharges"] + ["num_services"]

    lr_X_train, lr_X_val, lr_X_test, scaler = scale_numeric(
        lr_X_train, lr_X_val, lr_X_test, cols=lr_numeric_cols
    )
    logreg_set = {
        "X_train": lr_X_train, "X_val": lr_X_val, "X_test": lr_X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "scaler": scaler,
    }

    return logreg_set, xgb_set


if __name__ == "__main__":
    logreg_set, xgb_set = build_feature_sets()

    print("=== Logistic Regression feature set ===")
    print(f"Train shape: {logreg_set['X_train'].shape}")
    print(f"Columns: {logreg_set['X_train'].columns.tolist()}\n")

    print("=== XGBoost feature set ===")
    print(f"Train shape: {xgb_set['X_train'].shape}")
    print(f"Columns: {xgb_set['X_train'].columns.tolist()}")

    # Sanity check: scaled numeric cols in logreg set should have ~0 mean, ~1 std on TRAIN
    print("\n=== Scaling sanity check (train set, should be ~0 mean / ~1 std) ===")
    print(logreg_set["X_train"][["tenure", "MonthlyCharges"]].describe().loc[["mean", "std"]])