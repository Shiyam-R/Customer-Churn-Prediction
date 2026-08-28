"""
Phase 1 / Project 1 — Customer Churn: Model Selection

Trains two models on the two feature sets from feature_engineering.py:
- Logistic Regression (scaled features, TotalCharges dropped, class_weight='balanced')
- XGBoost (unscaled, TotalCharges kept, scale_pos_weight for imbalance)

Evaluated at default threshold (0.5) first as a baseline — threshold
tuning happens in a separate pass (eval_report.py) using the real
precision-recall curve, not a guessed number.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, average_precision_score,
    roc_auc_score, classification_report
)
from xgboost import XGBClassifier
from Churn_Feature_engineering import build_feature_sets


def train_logreg(logreg_set):
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(logreg_set["X_train"], logreg_set["y_train"])
    return model


def train_xgboost(xgb_set):
    neg = (xgb_set["y_train"] == 0).sum()
    pos = (xgb_set["y_train"] == 1).sum()
    scale_pos_weight = neg / pos
    print(f"scale_pos_weight = {neg}/{pos} = {scale_pos_weight:.3f}")

    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(xgb_set["X_train"], xgb_set["y_train"])
    return model


def evaluate(model, X_val, y_val, name: str, threshold: float = 0.5):
    proba = model.predict_proba(X_val)[:, 1]
    preds = (proba >= threshold).astype(int)

    print(f"\n=== {name} — Validation @ threshold {threshold} ===")
    print(f"Precision: {precision_score(y_val, preds):.3f}")
    print(f"Recall:    {recall_score(y_val, preds):.3f}")
    print(f"PR-AUC:    {average_precision_score(y_val, proba):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_val, proba):.3f}  (reference only — PR-AUC is the trustworthy one here)")
    print(classification_report(y_val, preds, target_names=["No Churn", "Churn"]))

    return proba


# Final operating threshold for XGBoost, chosen via cost-sensitive analysis
# (FN cost ~7x FP cost — losing a customer vs. an unnecessary retention email).
# Selected on the VALIDATION set only; test set stays untouched until the
# final eval report, per the train/val/test discipline established in
# Phase 1 block 2.
XGB_THRESHOLD = 0.34


if __name__ == "__main__":
    logreg_set, xgb_set = build_feature_sets()

    print("Training Logistic Regression...")
    logreg_model = train_logreg(logreg_set)

    print("\nTraining XGBoost...")
    xgb_model = train_xgboost(xgb_set)

    logreg_proba = evaluate(logreg_model, logreg_set["X_val"], logreg_set["y_val"], "Logistic Regression", threshold=0.5)
    xgb_proba = evaluate(xgb_model, xgb_set["X_val"], xgb_set["y_val"], "XGBoost @ default 0.5", threshold=0.5)
    xgb_proba_final = evaluate(xgb_model, xgb_set["X_val"], xgb_set["y_val"], "XGBoost @ chosen threshold", threshold=XGB_THRESHOLD)