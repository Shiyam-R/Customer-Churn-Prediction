"""
Phase 1 / Project 1 — Customer Churn: Hyperparameter Tuning (XGBoost)

XGBoost was selected in model_selection.py (better PR-AUC than logistic
regression). This tunes its hyperparameters.

Methodology: uses PredefinedSplit so the search fits on TRAIN and scores
on VAL only — no k-fold CV, staying consistent with the fixed train/val/test
split established in prepare_data.py. Test set is not touched here.

scale_pos_weight is held FIXED at the data-derived ratio (neg/pos) rather
than searched — it was already justified analytically in model_selection.py,
and searching it alongside other params risks re-litigating that decision
via a noisier signal.

After tuning, re-runs the cost-sensitive threshold analysis (FN cost ~7x FP
cost, per the earlier design decision) on the TUNED model's probabilities,
since tuning shifts what probabilities the model outputs — the threshold
chosen for the untuned model does not automatically transfer.
"""

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
from Churn_Feature_engineering import build_feature_sets

RANDOM_STATE = 42
FN_COST_RATIO = 7  # locked in from the earlier cost-sensitive threshold decision


def build_predefined_split(X_train, X_val):
    """
    test_fold: -1 for train rows (never held out), 0 for val rows (always
    held out and scored). This makes PredefinedSplit behave as a single
    fixed train/val split instead of k-fold.
    """
    test_fold = np.concatenate([
        np.full(len(X_train), -1),
        np.full(len(X_val), 0),
    ])
    return PredefinedSplit(test_fold)


def tune_xgboost(xgb_set):
    X_combined = np.vstack([xgb_set["X_train"].values, xgb_set["X_val"].values])
    y_combined = np.concatenate([xgb_set["y_train"].values, xgb_set["y_val"].values])
    ps = build_predefined_split(xgb_set["X_train"], xgb_set["X_val"])

    neg = (xgb_set["y_train"] == 0).sum()
    pos = (xgb_set["y_train"] == 1).sum()
    scale_pos_weight = neg / pos

    param_dist = {
        "n_estimators": [100, 150, 200, 300, 400],
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "min_child_weight": [1, 3, 5, 7],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.2, 0.5],
    }

    base_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=40,
        scoring="average_precision",  # PR-AUC — the trustworthy metric here
        cv=ps,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_combined, y_combined)

    print(f"\nBest PR-AUC (on val, via search): {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    return search.best_estimator_, search.best_params_


def find_optimal_threshold(model, X_val, y_val, fn_cost=FN_COST_RATIO):
    proba = model.predict_proba(X_val)[:, 1]
    y_val = np.asarray(y_val)
    thresholds = np.arange(0.05, 0.95, 0.005)

    best_thresh, best_cost = None, float("inf")
    for t in thresholds:
        preds = (proba >= t).astype(int)
        fn = ((preds == 0) & (y_val == 1)).sum()
        fp = ((preds == 1) & (y_val == 0)).sum()
        total_cost = fn * fn_cost + fp * 1
        if total_cost < best_cost:
            best_cost = total_cost
            best_thresh = t

    preds = (proba >= best_thresh).astype(int)
    tp = ((preds == 1) & (y_val == 1)).sum()
    fp = ((preds == 1) & (y_val == 0)).sum()
    fn = ((preds == 0) & (y_val == 1)).sum()
    recall = tp / (tp + fn)
    precision = tp / max((preds == 1).sum(), 1)

    print(f"\n=== Tuned model — optimal threshold (FN cost={fn_cost}x) ===")
    print(f"threshold={best_thresh:.3f}, recall={recall:.3f}, precision={precision:.3f}, "
          f"TP={tp}, FP={fp}, FN={fn}")

    return best_thresh


if __name__ == "__main__":
    logreg_set, xgb_set = build_feature_sets()

    print("Tuning XGBoost (fit on train, scored on val, no k-fold)...")
    tuned_model, best_params = tune_xgboost(xgb_set)

    # Compare against the untuned baseline PR-AUC (0.649, from model_selection.py)
    val_proba = tuned_model.predict_proba(xgb_set["X_val"])[:, 1]
    tuned_pr_auc = average_precision_score(xgb_set["y_val"], val_proba)
    print(f"\nTuned model PR-AUC on val: {tuned_pr_auc:.4f}  (baseline was 0.649)")

    optimal_threshold = find_optimal_threshold(tuned_model, xgb_set["X_val"], xgb_set["y_val"])