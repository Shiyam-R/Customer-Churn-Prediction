"""
Phase 1 / Project 1 — Customer Churn: Final Evaluation Report

ONE-TIME test set evaluation. Model trained on TRAIN ONLY (not retrained
on train+val — deliberate choice: preserves a clean recovery path if this
evaluation reveals a problem, at the cost of not using val's extra rows
in the final model).

Model: XGBoost, tuned hyperparameters from hyperparameter_tuning.py
Threshold: 0.285, chosen via cost-sensitive analysis (FN cost ~7x FP cost)
on the validation set — never touched using test data.
"""

import json
from datetime import datetime, timezone

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, average_precision_score,
    roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier
import shap
from Churn_Feature_engineering import build_feature_sets

FINAL_THRESHOLD = 0.285
MODEL_VERSION = "1.0.0"
BEST_PARAMS = {
    "subsample": 0.8, "n_estimators": 300, "min_child_weight": 3,
    "max_depth": 3, "learning_rate": 0.03, "gamma": 0.1, "colsample_bytree": 0.7,
}


def train_final_model(xgb_set):
    neg = (xgb_set["y_train"] == 0).sum()
    pos = (xgb_set["y_train"] == 1).sum()
    scale_pos_weight = neg / pos

    model = XGBClassifier(
        **BEST_PARAMS,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(xgb_set["X_train"], xgb_set["y_train"])
    return model


def run_test_evaluation(model, X_test, y_test):
    """The ONE-TIME test set evaluation. Run once, report as-is — no
    tuning based on these numbers, per the leakage discipline from block 2."""
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= FINAL_THRESHOLD).astype(int)

    print("=" * 60)
    print("FINAL TEST SET EVALUATION (one-time)")
    print("=" * 60)
    print(f"Precision: {precision_score(y_test, preds):.3f}")
    print(f"Recall:    {recall_score(y_test, preds):.3f}")
    print(f"F1:        {f1_score(y_test, preds):.3f}")
    print(f"PR-AUC:    {average_precision_score(y_test, proba):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, proba):.3f}  (reference only)")
    print("\n" + classification_report(y_test, preds, target_names=["No Churn", "Churn"]))

    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Pred: No Churn", "Pred: Churn"],
                yticklabels=["Actual: No Churn", "Actual: Churn"])
    ax.set_title(f"Test Set Confusion Matrix (threshold={FINAL_THRESHOLD})")
    plt.tight_layout()
    plt.savefig("test_confusion_matrix.png", dpi=120)
    plt.close()
    print("\nSaved test_confusion_matrix.png")

    return proba, preds


def run_shap_analysis(model, X_train, X_test):
    """SHAP explainability — Project 1's required deliverable."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    fig = plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, X_test, show=False, plot_size=None)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved shap_summary.png")

    # Mean absolute SHAP value per feature = global importance ranking
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance = sorted(zip(X_test.columns, mean_abs_shap), key=lambda x: -x[1])
    print("\n=== Top 10 features by mean |SHAP value| ===")
    for feat, val in importance[:10]:
        print(f"{feat}: {val:.4f}")

    return shap_values


def compute_baseline_stats(X_train, n_bins: int = 10, low_cardinality_threshold: int = 10) -> dict:
    """
    Per-feature binned baseline distribution for /drift's PSI calculation.

    Low-cardinality columns (binary 0/1, ordinal Contract 0/1/2,
    num_services 0-6) are binned on their EXACT unique values, not
    quantiles — quantile binning (qcut) degenerates to a single bin on
    skewed or few-valued data (verified: it did exactly this for `gender`
    before this fix, silently making drift detection useless for most
    columns). Genuinely continuous columns (tenure, MonthlyCharges,
    TotalCharges) use quantile bins as intended.
    """
    import pandas as pd

    stats = {}
    for col in X_train.columns:
        n_unique = X_train[col].nunique()

        if n_unique <= low_cardinality_threshold:
            unique_vals = sorted(X_train[col].unique())
            edges = [unique_vals[0] - 0.5] + [v + 0.5 for v in unique_vals]
        else:
            try:
                _, qcut_edges = pd.qcut(X_train[col], q=n_bins, duplicates="drop", retbins=True)
            except ValueError:
                continue
            edges = qcut_edges.tolist()

        binned = pd.cut(X_train[col], bins=edges, include_lowest=True)
        proportions = binned.value_counts(normalize=True, sort=False).to_numpy().tolist()
        stats[col] = {"bin_edges": edges, "baseline_proportions": proportions}
    return stats


if __name__ == "__main__":
    logreg_set, xgb_set = build_feature_sets()

    print("Training final model (train only, tuned hyperparameters)...\n")
    final_model = train_final_model(xgb_set)

    proba, preds = run_test_evaluation(final_model, xgb_set["X_test"], xgb_set["y_test"])

    print("\n" + "=" * 60)
    print("SHAP Explainability")
    print("=" * 60)
    run_shap_analysis(final_model, xgb_set["X_train"], xgb_set["X_test"])

    # --- Persist the model artifact for the API ---
    # Uses XGBoost's NATIVE save format, not pickle/joblib. Pickle serializes
    # internal version-check metadata that isn't guaranteed compatible across
    # environments (dev sandbox -> Docker -> Render each count as a new
    # environment) — XGBoost's own docs recommend save_model()/load_model()
    # specifically to avoid this. See the UserWarning this used to produce:
    # "please export the model by calling Booster.save_model() ... first"
    final_model.save_model("churn_model.json")
    with open("model_columns.json", "w") as f:
        json.dump(xgb_set["X_train"].columns.tolist(), f)

    # Model metadata — powers /version's "when was the current model
    # trained" reporting. GIT_SHA/build identity is injected separately,
    # at Docker build time (see Dockerfile + ci.yml), not here — this
    # script only knows about the MODEL, not the eventual deployment.
    test_recall = recall_score(xgb_set["y_test"], preds)
    test_precision = precision_score(xgb_set["y_test"], preds)
    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "test_recall": round(float(test_recall), 4),
        "test_precision": round(float(test_precision), 4),
        "threshold": FINAL_THRESHOLD,
    }
    with open("model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Baseline feature distributions — powers /drift's PSI computation.
    baseline_stats = compute_baseline_stats(xgb_set["X_train"])
    with open("baseline_stats.json", "w") as f:
        json.dump(baseline_stats, f)

    print("\nSaved churn_model.json, model_columns.json, model_metadata.json, "
          "and baseline_stats.json for API use.")