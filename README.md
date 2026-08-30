# Customer Churn Prediction API

[![CI Pipeline](https://github.com/Shiyam-R/Customer-Churn-Prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Shiyam-R/Customer-Churn-Prediction/actions/workflows/ci.yml)  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**🔗 Live demo: https://customer-churn-prediction-xrp5.onrender.com/docs**

A production-quality customer churn classification system built on the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset — from leakage-aware feature engineering and cost-sensitive model optimization through a containerized, CI-tested, SHAP-explainable FastAPI inference service.

## Overview

Given a customer's account and service attributes, the API returns a churn probability, a threshold-based prediction, and the top factors driving that specific prediction using per-request SHAP explanations.

The final model is a tuned XGBoost classifier:

| Component              | Detail                                      |
| ---------------------- | ------------------------------------------- |
| Model                  | XGBoost                                     |
| Hyperparameter tuning  | RandomizedSearchCV, 40 candidates           |
| Best validation PR-AUC | `0.6930`                                    |
| Decision threshold     | `0.345`                                     |
| Threshold strategy     | Cost-sensitive optimization                 |
| False-negative cost    | Assumed 7× higher than a false positive     |
| Explainability         | Per-request SHAP using `shap.TreeExplainer` |
| Data split             | Stratified 70/20/10 train/validation/test   |

The classification threshold isn't left at the default `0.5`. It was selected using a cost-sensitive threshold sweep where missing a real churner is assumed to be approximately 7× more costly than an unnecessary retention action.

The final model achieved the following results on the untouched test set:

| Metric    |   Score |
| --------- | ------: |
| Precision | `0.473` |
| Recall    | `0.877` |
| F1 Score  | `0.614` |
| PR-AUC    | `0.687` |
| ROC-AUC   | `0.857` |

The selected threshold provides a more balanced precision-recall trade-off than the earlier lower threshold while still prioritizing the detection of customers likely to churn.

### What makes this more than a training script

* **Leakage traced to mechanism, not just correlation.** An engineered feature (`avg_charge_deviation`, comparing current billing to a customer's historical average) initially appeared promising until its distribution was traced to a mechanical relationship with `tenure`. The feature was rejected after understanding why it behaved predictively rather than relying only on correlation or feature importance.
* **Interaction effects hunted deliberately, then verified with SHAP.** EDA identified that `Contract=Month-to-month` combined with `PaymentMethod=Electronic check` had a particularly high churn rate. This interaction was encoded as `high_risk_combo`, which was subsequently confirmed by SHAP as one of the model's top features.
* **Evaluation discipline enforced structurally.** Stratified 70/20/10 train/validation/test splits preserved the ~26.5% churn rate across all datasets. Model selection, hyperparameter tuning, and threshold optimization were performed using training and validation data, while the test set was evaluated once at the end.
* **Cost-sensitive decision making.** The classification threshold was optimized around business cost rather than accepting the conventional `0.5` threshold. With false negatives assumed to be 7× more expensive, the final threshold prioritizes churn detection while improving the precision-recall trade-off.

## Model Performance

### Hyperparameter Tuning

XGBoost was tuned using `RandomizedSearchCV` with 40 candidate configurations, trained on the training split and evaluated using PR-AUC on the validation set.

```text
Best PR-AUC (validation): 0.6930

Best parameters:
n_estimators: 300
min_child_weight: 3
max_depth: 3
learning_rate: 0.03
gamma: 0.1
subsample: 0.8
colsample_bytree: 0.7
```

The tuned model improved validation PR-AUC from approximately `0.649` for the baseline configuration to `0.693`.

### Threshold Optimization

The optimal threshold was selected using a cost-sensitive sweep with a false-negative cost assumed to be 7× higher than a false-positive cost.

```text
Threshold: 0.345
Recall:    0.928
Precision: 0.471

TP: 347
FP: 390
FN: 27
```

### Final Test Evaluation

The final model was evaluated once on the untouched test set:

```text
Precision: 0.473
Recall:    0.877
F1:        0.614
PR-AUC:    0.687
ROC-AUC:   0.857
```

Classification report:

```text
              precision    recall  f1-score   support

    No Churn       0.94      0.65      0.76       518
       Churn       0.47      0.88      0.61       187

    accuracy                           0.71       705
   macro avg       0.70      0.76      0.69       705
weighted avg       0.81      0.71      0.72       705
```

The model achieves **87.7% recall for churners**, meaning it identifies the majority of customers who actually churn.

The higher threshold improves precision compared with the earlier `0.285` operating threshold while reducing recall slightly. This represents the expected precision-recall trade-off from selecting a stricter classification boundary.

The evaluation pipeline also generates a confusion matrix and SHAP summary plot.

### SHAP Explainability

Top features by mean absolute SHAP value:

| Rank | Feature                       | Mean |SHAP value| |
| ---- | ----------------------------- | ----------------: |
| 1    | `Contract`                    |          `0.6813` |
| 2    | `tenure`                      |          `0.4848` |
| 3    | `InternetService_Fiber optic` |          `0.2971` |
| 4    | `MonthlyCharges`              |          `0.2182` |
| 5    | `TotalCharges`                |          `0.1716` |
| 6    | `high_risk_combo`             |          `0.1571` |
| 7    | `PaperlessBilling`            |          `0.1230` |
| 8    | `InternetService_No`          |          `0.1227` |
| 9    | `OnlineSecurity`              |          `0.1145` |
| 10   | `StreamingTV`                 |          `0.1063` |

`Contract` and `tenure` are the strongest drivers of churn predictions, while the engineered `high_risk_combo` also ranks among the top features, supporting the interaction identified during EDA.

## Project Structure

```text
Customer-Churn-Prediction/
├── app/
│   ├── main.py                  # FastAPI application factory + lifespan
│   ├── config.py                # threshold, rate limit, artifact paths
│   ├── exceptions.py            # custom exception hierarchy
│   ├── model_loader.py          # artifact loading singleton
│   ├── rate_limiter.py          # slowapi Limiter instance
│   ├── drift_tracker.py         # in-memory live-request buffer + PSI computation
│   ├── api/
│   │   └── routes.py            # /, /health, /version, /drift, /predict
│   ├── schemas/
│   │   ├── request.py
│   │   └── response.py
│   ├── services/
│   │   └── prediction_service.py
│   └── utils/
│       ├── feature_engineering.py
│       └── logger.py
├── artifacts/                   # churn_model.json, model_columns.json, model_metadata.json, baseline_stats.json
├── data/                        # raw and processed data
├── EDA_plots/                   # exploratory data analysis visualizations
├── Eval_plots/                  # confusion matrix and SHAP summary
├── notebook/                    # pipeline scripts: preprocessing → EDA → feature engineering → model selection → tuning → evaluation
├── tests/                       # pytest suite
├── load_tests/                  # Locust load testing + measured results
├── .github/workflows/ci.yml     # CI: lint, test, security audit, Docker build, publish + deploy
├── Dockerfile                   # multi-stage production image
├── pytest.ini                   # pytest configuration
├── requirements.txt             # production dependencies
├── requirements-dev.txt         # testing and development dependencies
├── requirements-load.txt        # Locust dependencies
├── .dockerignore
├── .gitignore
├── LICENSE
└── README.md
```

Training scripts and unnecessary development files are excluded from the Docker image through `.dockerignore` — the serving container only needs the application code and trained artifacts required for inference.

## Getting Started

### Prerequisites

* Python 3.12
* Docker (optional, for containerized runs)

### Local setup

```bash
git clone https://github.com/Shiyam-R/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# Production + testing dependencies
pip install -r requirements-dev.txt
```

### Run locally

```bash
uvicorn app.main:app --reload
```

* Swagger UI: http://localhost:8000/docs
* ReDoc: http://localhost:8000/redoc

### Run with Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## API Reference

| Method | Endpoint   | Purpose                                                                             |
| ------ | ---------- | ----------------------------------------------------------------------------------- |
| `GET`  | `/`        | Project metadata and available endpoints                                            |
| `GET`  | `/health`  | Reports whether the model artifact loaded successfully                              |
| `GET`  | `/version` | Reports API/model version, training metadata, build git SHA, and environment        |
| `GET`  | `/drift`   | Feature-distribution drift using PSI between recent live requests and training data |
| `POST` | `/predict` | Returns churn prediction and per-request SHAP explanation                           |

The full request/response schema — including every field, its valid values, and validation rules — is generated automatically from the Pydantic models and browsable at `/docs`.

Example request for `POST /predict`:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.0,
  "TotalCharges": 170.0
}
```

Example response:

```json
{
  "prediction": "Churn",
  "churn_probability": 0.7845,
  "threshold_used": 0.345,
  "top_contributing_factors": [
    {
      "feature": "Contract",
      "shap_value": 0.5083
    },
    {
      "feature": "tenure",
      "shap_value": 0.5070
    },
    {
      "feature": "InternetService_Fiber optic",
      "shap_value": -0.3501
    },
    {
      "feature": "TotalCharges",
      "shap_value": 0.2511
    },
    {
      "feature": "high_risk_combo",
      "shap_value": 0.2265
    }
  ]
}
```

Categorical fields are `Literal`-constrained in the Pydantic schema — an unseen value such as `PaymentMethod: "Cryptocurrency"` is rejected with a `422` validation error before reaching the model.

## Configuration

Configuration currently lives in `app/config.py`, with build-time values injected where required:

| Setting              | Value                        | Purpose                                 |
| -------------------- | ---------------------------- | --------------------------------------- |
| `THRESHOLD`          | `0.345`                      | Cost-sensitive classification threshold |
| `PREDICT_RATE_LIMIT` | `30/minute`                  | Per-client limit on `/predict`          |
| `GIT_SHA`            | Docker build argument        | Injected during CI builds               |
| `ENVIRONMENT`        | `production` / `development` | Deployment environment                  |

The `/predict` rate limit helps protect the API from resource exhaustion because SHAP explanations introduce additional computation for every request.

## Testing

```bash
pytest
```

The test suite covers feature engineering and training-serving consistency, schema validation, unseen-category rejection, service-layer error paths, API-level integration, rate limiting, and the `/`, `/version`, and `/drift` endpoints.

## Load Testing

Concurrency and latency testing lives in `load_tests/` (Locust) — separate from `tests/` since it exercises a live running instance rather than mocked components. See [`load_tests/LOAD_TEST_RESULTS.md`](load_tests/LOAD_TEST_RESULTS.md) for setup, measured results, and interpretation.

## CI/CD

Every push to `main` runs `.github/workflows/ci.yml`:

1. Checkout → Python 3.12 setup
2. Install `requirements.txt` and `requirements-dev.txt`
3. Lint with `ruff`
4. Run the `pytest` suite
5. Run dependency vulnerability scanning with `pip-audit`
6. Build the production Docker image and verify the container starts successfully
7. Publish the validated image
8. Trigger a Render deployment

The container is started and checked after building because a successful Docker build alone does not guarantee that the application actually starts correctly.

## Logging

Structured logs are written to the application console through `app/utils/logger.py`, making them suitable for containerized deployment platforms such as Render.

## License

[MIT](LICENSE)