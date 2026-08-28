"""
Phase 1 / Project 1 — Customer Churn: EDA (deep pass + univariate sweep)

Imports load/clean/split from prepare_data.py — this file does EDA only,
no duplicated data logic. Run after prepare_data.py's telco_churn.csv is
in the same folder.

Covers:
1. Categorical churn-rate breakdowns (contract, tenure, payment, internet)
2. Interaction check: Contract x PaymentMethod (tests whether PaymentMethod's
   signal is redundant with Contract, or independent — it's independent)
3. Distribution plots: MonthlyCharges/tenure split by churn
4. Correlation heatmap for numeric features
5. Univariate sweep — churn rate per category for every column not
   deep-dived above (coverage/data-quality check, not a substitute for
   post-model SHAP/permutation importance)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prepare_data import load_and_clean, stratified_split

sns.set_style("whitegrid")


def run_eda(df: pd.DataFrame):
    df = df.copy()
    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=[-1, 6, 12, 24, 48, 100],
        labels=["0-6mo", "6-12mo", "1-2yr", "2-4yr", "4yr+"]
    )

    # --- Categorical churn-rate breakdowns ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, col, title in zip(
        axes.flat,
        ["Contract", "tenure_bucket", "PaymentMethod", "InternetService"],
        ["Churn Rate by Contract", "Churn Rate by Tenure Bucket",
         "Churn Rate by Payment Method", "Churn Rate by Internet Service"],
    ):
        rates = df.groupby(col, observed=True)["Churn"].mean().sort_values(ascending=False)
        rates.plot(kind="bar", ax=ax, color="#4C72B0")
        ax.set_title(title)
        ax.set_ylabel("Churn Rate")
        ax.axhline(df["Churn"].mean(), color="red", linestyle="--", linewidth=1, label="Overall rate")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig("eda_categorical_breakdowns.png", dpi=120)
    plt.close()
    print("Saved eda_categorical_breakdowns.png")

    # --- Interaction check: Contract x PaymentMethod ---
    interaction = df.pivot_table(values="Churn", index="Contract", columns="PaymentMethod", aggfunc="mean")
    print("\n=== Churn rate: Contract x PaymentMethod ===")
    print(interaction.round(3))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.heatmap(interaction, annot=True, fmt=".2f", cmap="Reds", ax=ax, cbar_kws={"label": "Churn rate"})
    ax.set_title("Churn Rate: Contract x PaymentMethod (interaction check)")
    plt.tight_layout()
    plt.savefig("eda_interaction_heatmap.png", dpi=120)
    plt.close()
    print("Saved eda_interaction_heatmap.png")

    # --- Distributions split by churn ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.kdeplot(data=df, x="MonthlyCharges", hue="Churn", fill=True, alpha=0.4, ax=axes[0])
    axes[0].set_title("MonthlyCharges Distribution by Churn")
    sns.kdeplot(data=df, x="tenure", hue="Churn", fill=True, alpha=0.4, ax=axes[1])
    axes[1].set_title("Tenure Distribution by Churn")
    plt.tight_layout()
    plt.savefig("eda_distributions.png", dpi=120)
    plt.close()
    print("Saved eda_distributions.png")

    # --- Correlation heatmap ---
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Numeric Feature Correlation")
    plt.tight_layout()
    plt.savefig("eda_correlation_heatmap.png", dpi=120)
    plt.close()
    print("Saved eda_correlation_heatmap.png")

    # --- Diagnostic check: does the MonthlyCharges bimodal peak map to InternetService? ---
    print("\n=== MonthlyCharges by InternetService (checks the cheap/expensive peak split) ===")
    print(df.groupby("InternetService")["MonthlyCharges"].describe()[["mean", "50%", "std", "count"]])

    print("\nEDA complete. 4 charts saved.")


def univariate_sweep(df: pd.DataFrame):
    """
    Lightweight coverage check — churn rate per category, for every
    categorical column NOT already deep-dived in run_eda() (Contract,
    tenure, PaymentMethod, InternetService, MonthlyCharges/TotalCharges).

    Purpose: catch data-quality issues (bad encodings, unexpected category
    values) and any standout univariate signal BEFORE modeling — not a
    substitute for post-model SHAP/permutation importance, which evaluates
    features in the context of the full model. This is a coverage/hygiene
    pass, not a feature-hunting exercise.
    """
    already_covered = {"Contract", "tenure", "PaymentMethod", "InternetService",
                        "MonthlyCharges", "TotalCharges", "Churn", "customerID"}
    remaining_cols = [c for c in df.columns if c not in already_covered]

    print("\n" + "=" * 60)
    print("Univariate sweep — remaining columns (coverage check)")
    print("=" * 60)

    overall_rate = df["Churn"].mean()
    for col in remaining_cols:
        rates = df.groupby(col)["Churn"].agg(["mean", "count"])
        rates["lift_vs_overall"] = (rates["mean"] - overall_rate).round(3)
        print(f"\n--- {col} ---")
        print(rates.round(3))


# =================================================================
# MAIN
# =================================================================

if __name__ == "__main__":
    df = load_and_clean()
    print(f"\nShape after cleaning: {df.shape}")
    print(f"Overall churn rate: {df['Churn'].mean():.3f}\n")

    # Split is run here too so this script mirrors the same leakage-safe
    # split prepare_data.py defines — not duplicated logic, same function.
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(df)

    print("\n" + "=" * 60)
    print("Running EDA...")
    print("=" * 60 + "\n")
    run_eda(df)

    univariate_sweep(df)