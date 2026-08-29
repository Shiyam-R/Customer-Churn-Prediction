# Customer Churn Prediction API

[![CI Pipeline](https://github.com/Shiyam-R/Customer-Churn-Prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Shiyam-R/Customer-Churn-Prediction/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**🔗 Live demo:** _pending Render deployment — update this line once the service is live._

A production-quality churn classification system built on the [Telco Customer
Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
dataset — from leakage-aware feature engineering through a containerized,
CI-tested, SHAP-explainable FastAPI inference service.

---

## Overview

Given a customer's account and service attributes, the API returns a churn
probability, a threshold-based label, and the top factors (via per-request
SHAP) driving that specific prediction — not a static importance list.

| Component | Detail |
|---|---|
| Model | XGBoost, tuned (`n_estimators=300, max_depth=3, learning_rate=0.03, ...`) |
| Decision threshold | **0.285**, not the default 0.5 |
| Explainability | Per-request SHAP (`shap.TreeExplainer`), not a fixed feature list |

The threshold isn't a convention — it's the output of a cost-sensitive
sweep (missing a real churner assumed ~7x costlier than an unnecessary
retention email), evaluated against the actual precision/recall tradeoff
curve for this specific tuned model.

### What makes this more than a training script

- **Leakage traced to mechanism, not just correlation.** An engineered
  feature (`avg_charge_deviation`, comparing current billing to a
  customer's historical average) looked promising until its entire
  distribution — both the near-zero center *and* the U-shaped tails — was
  traced back to being a mechanical proxy for `tenure` (an average over
  more billing months mechanically has less variance, independent of any
  real pricing behavior). Rejected once the mechanism was understood, not
  just because a summary statistic looked weak.
- **Interaction effects hunted deliberately, then verified against SHAP.**
  EDA surfaced that `Contract=Month-to-month` combined with
  `PaymentMethod=Electronic check` churns at 54% — well above what either
  factor alone, or their additive combination, would predict. Encoded as
  `high_risk_combo`, and confirmed post-hoc: it ranks in the model's top 5
  SHAP features, ahead of `TotalCharges` and most raw columns.
- **Evaluation discipline enforced structurally, not just by habit.**
  Stratified 70/20/10 train/val/test split (preserving the ~26.5% churn
  rate across all three); every model-selection, tuning, and threshold
  decision made on validation; the test set touched exactly once, at the
  end, producing 92.0% recall / 44.1% precision — closely matching
  validation numbers, the actual evidence the model generalizes rather
  than having overfit to repeated validation-set tuning.

## Project Structure

```
Customer-Churn-Prediction/
├── app/
│   ├── main.py                      # FastAPI application factory + lifespan
│   ├── config.py                    # threshold, rate limit, artifact paths
│   ├── exceptions.py                # custom exception hierarchy
│   ├── model_loader.py              # artifact loading singleton
│   ├── rate_limiter.py              # slowapi Limiter instance
│   ├── api/routes.py                # /predict, /health
│   ├── schemas/{request,response}.py
│   ├── services/prediction_service.py
│   └── utils/{feature_engineering,logger}.py
├── artifacts/                       # churn_model.json (native XGBoost format), model_columns.json
├── tests/                           # pytest suite (21 tests)
├── load_tests/                      # Locust + measured results
├── reports/                         # EDA charts, PR curve, confusion matrices, SHAP summary
├── prepare_data.py                  # load, clean, stratified split
├── eda_only.py                      # EDA + univariate sweep
├── feature_engineering.py           # training-time encoding + engineered features
├── model_selection.py               # logistic regression vs XGBoost baseline
├── hyperparameter_tuning.py         # RandomizedSearchCV on XGBoost
├── eval_report.py                   # one-time test evaluation + SHAP + saves artifacts
├── .github/workflows/ci.yml         # lint, test, security audit, Docker build, publish + deploy
├── Dockerfile                       # multi-stage production image
├── requirements.txt                 # production dependencies
├── requirements-dev.txt             # + pytest, httpx2
└── requirements-load.txt            # + locust
```

Training scripts and the raw dataset are excluded from the Docker image
(see `.dockerignore`) — the serving container only ever needs `app/` and
`artifacts/`, never the training pipeline or its dependencies.

## Getting Started

### Prerequisites

- Python 3.12
- Docker (optional, for containerized runs)

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

- Swagger UI: <http://localhost:8000/docs>
- ReDoc: <http://localhost:8000/redoc>

### Run with Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## API Reference

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/predict` | Returns churn prediction + per-request SHAP explanation |
| `GET` | `/health` | Reports whether the model artifact loaded successfully |

The full request/response schema — including every field, its valid
values, and validation rules — is generated automatically from the
Pydantic models and browsable at `/docs`. A real captured example:

```json
{
  "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
  "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
  "MultipleLines": "No", "InternetService": "Fiber optic",
  "OnlineSecurity": "No", "OnlineBackup": "No",
  "DeviceProtection": "No", "TechSupport": "No",
  "StreamingTV": "No", "StreamingMovies": "No",
  "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.0, "TotalCharges": 170.0
}
```

```json
{
  "prediction": "Churn",
  "churn_probability": 0.7712,
  "threshold_used": 0.285,
  "top_contributing_factors": [
    {"feature": "Contract", "shap_value": 0.5265},
    {"feature": "tenure", "shap_value": 0.5164},
    {"feature": "InternetService_Fiber optic", "shap_value": -0.3546},
    {"feature": "TotalCharges", "shap_value": 0.2424},
    {"feature": "high_risk_combo", "shap_value": 0.2003}
  ]
}
```

Categorical fields are `Literal`-constrained in the Pydantic schema — an
unseen value (e.g. `PaymentMethod: "Cryptocurrency"`) is rejected with a
`422` before it ever reaches the model, rather than silently mispredicted.

## Configuration

Currently hardcoded in `app/config.py` — no environment-variable overrides
implemented yet (unlike a 12-factor setup with a `.env.example`):

| Setting | Value | Purpose |
|---|---|---|
| `THRESHOLD` | `0.285` | Cost-sensitive decision threshold |
| `PREDICT_RATE_LIMIT` | `30/minute` | Per-client limit on `/predict` (SHAP runs per request — real compute cost) |

## Testing

```bash
pytest
```

21 tests: encoding/feature logic (including the training-serving skew
guardrail), schema validation (including the unseen-category rejection),
service-layer error paths, and API-level integration tests including the
rate limiter actually triggering under load.

## Security

```bash
pip install pip-audit
pip-audit -r requirements.txt
```

Runs automatically in CI on every push. `/predict`'s rate limit (above) is
the other half of the security story — mitigating resource-exhaustion
abuse given the real per-request SHAP cost.

## Load Testing

Concurrency and latency testing lives in `load_tests/` (Locust) — separate
from `tests/` since it exercises a live running instance rather than
mocked components. See
[`load_tests/LOAD_TEST_RESULTS.md`](load_tests/LOAD_TEST_RESULTS.md) for
setup, real measured numbers, and an honest discussion of what a
single-source-IP test does and doesn't demonstrate about true multi-client
capacity.

## CI/CD

Every push to `main` runs `.github/workflows/ci.yml`:

1. Checkout → Python 3.12 setup
2. Install `requirements.txt` + `requirements-dev.txt`
3. Lint (`ruff`)
4. Run the `pytest` suite
5. Dependency vulnerability scan (`pip-audit`)
6. Build the production Docker image, run it, and poll `/health` — a
   successful build alone isn't proof the container actually starts
7. Publish the image to GHCR (`:latest` and `:<commit-sha>`)
8. Trigger a Render deploy hook — Render deploys the exact image CI just
   validated, rather than rebuilding separately

## Logging

Structured console logging via `app/utils/logger.py` (no file-based
rotation implemented — console output only, unlike a `logs/` + rotating
file handler setup).

## License

[MIT](LICENSE)