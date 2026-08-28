"""
Phase 1 / Project 1 — Customer Churn: Data Prep

Design decisions (per our discussion):
- Target: Churn (binary)
- Split: stratified 70/20/10 train/val/test — no time-series structure here,
  so chronological split isn't needed, but churn is imbalanced (~26%) so we
  stratify to keep that ratio consistent across all three splits.
- TotalCharges: NOT leakage (it's cumulative billing up to current status,
  not post-outcome data). It IS collinear with tenure * MonthlyCharges.
  We keep both raw versions available and let the modeling step decide
  per-model (drop for logistic regression, keep for XGBoost).
"""

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/Telco-Customer-Churn.csv"
RANDOM_STATE = 42


def load_and_clean(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # TotalCharges is read as object because a handful of rows have blank
    # strings (customers with tenure=0, i.e. brand new — never billed yet).
    # This is a real data quality issue, not a leakage issue — worth noting
    # in the eval report.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing = df["TotalCharges"].isna().sum()
    print(f"Rows with unparseable TotalCharges (likely tenure=0 new customers): {n_missing}")

    # For tenure=0 customers, TotalCharges is legitimately 0 (no billing yet) —
    # not a value to impute with mean/median, since that would fabricate
    # a number that misrepresents "hasn't been billed yet."
    df.loc[df["tenure"] == 0, "TotalCharges"] = 0
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Target: encode Yes/No -> 1/0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # customerID carries no predictive signal — drop it.
    df = df.drop(columns=["customerID"])

    return df


def stratified_split(df: pd.DataFrame, target_col: str = "Churn"):
    """
    70/20/10 train/val/test, stratified on target to preserve the
    ~26% churn rate across all three splits.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split off test (10%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=RANDOM_STATE
    )
    # Then split remaining 90% into train (70% of total) / val (20% of total)
    # val_size relative to X_temp = 0.20 / 0.90
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=(0.20 / 0.90), stratify=y_temp, random_state=RANDOM_STATE
    )

    for name, y_split in [("train", y_train), ("val", y_val), ("test", y_test)]:
        rate = y_split.mean()
        print(f"{name}: n={len(y_split)}, churn_rate={rate:.3f}")

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    df = load_and_clean()
    print(f"\nShape after cleaning: {df.shape}")
    print(f"Overall churn rate: {df['Churn'].mean():.3f}\n")

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(df)