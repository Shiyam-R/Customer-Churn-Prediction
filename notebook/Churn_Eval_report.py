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

import joblib
import json
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
    # Saves the model + its exact training column list (order matters —
    # see prepare_inference_features in feature_engineering.py for why).
    joblib.dump(final_model, "artifacts/churn_model.pkl")
    with open("artifacts/model_columns.json", "w") as f:
        json.dump(xgb_set["X_train"].columns.tolist(), f)
    print("\nSaved churn_model.pkl and model_columns.json for API use.")