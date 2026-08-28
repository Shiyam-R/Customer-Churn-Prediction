# Customer Churn Prediction

Phase 1 project: XGBoost churn classifier (Telco Customer Churn dataset),
served via FastAPI with per-request SHAP explanations.

## Pipeline (run in order)

```
python prepare_data.py           # load, clean, stratified 70/20/10 split
python eda_only.py                # EDA + univariate sweep
python feature_engineering.py     # encoding, scaling, engineered features
python model_selection.py         # logistic regression vs XGBoost baseline
python hyperparameter_tuning.py   # RandomizedSearchCV on XGBoost
python eval_report.py             # one-time test evaluation + SHAP
                                   # (also saves artifacts/churn_model.pkl
                                   #  and artifacts/model_columns.json)
```

## Model

- XGBoost, tuned (`n_estimators=300, max_depth=3, learning_rate=0.03, ...`)
- Decision threshold: **0.285** (not 0.5) — chosen via cost-sensitive
  analysis assuming a missed churner costs ~7x an unnecessary retention email
- Test set: 92.0% recall, 44.1% precision (never touched until the final
  one-time evaluation)

## API

```
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Docs at `http://127.0.0.1:8000/docs`. See `test_api.py` for a plain example
request. `/predict` is rate-limited (30/minute per client — SHAP runs on
every request, so this protects against resource-exhaustion abuse, not
just normal traffic).

Layered structure: `app/main.py` (app factory + lifespan startup) →
`app/api/routes.py` → `app/services/prediction_service.py` →
`app/model_loader.py` (artifacts loaded once at startup) +
`app/utils/feature_engineering.py` (inference-only, no training dependency).

## Testing

```
pip install -r requirements-dev.txt
pytest tests/ -v
```

21 tests: encoding/feature logic, schema validation (including the unseen-
category bug caught during development), service-layer error paths, and
API-level integration tests including the rate limiter.

## Security

```
pip install pip-audit
pip-audit -r requirements.txt
```

Also run automatically in CI on every push. `/predict` is rate-limited
(see API section above) as a resource-exhaustion mitigation, since each
request runs real SHAP computation, not just `predict_proba()`.

## Load Testing

```
pip install -r requirements-load.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000   # separate terminal
locust -f load_tests/locustfile.py --host http://localhost:8000 \
    --headless -u 20 -r 5 -t 30s
```

See `load_tests/LOAD_TEST_RESULTS.md` for real measured results and an
honest discussion of what they do/don't show (rate limiter behavior under
single-source load vs. true multi-client capacity).

## Docker

```
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## CI/CD

Push to `main` triggers: lint (ruff) → tests (pytest) → dependency audit
(pip-audit) → Docker build + health check → publish to GHCR
(`ghcr.io/<owner>/<repo>:latest` and `:<commit-sha>`) → trigger a Render
deploy hook. Render deploys the exact image CI already validated, rather
than rebuilding separately — one built-and-tested artifact, not two
pipelines that could drift apart.

**Manual one-time setup required** (not automatable from here):
1. On GitHub: after the first successful CD run, go to the repo's
   **Packages** tab, find the published image, and set its visibility to
   **public** (or configure a registry credential in Render) — GHCR images
   are private by default and Render can't pull a private one without auth.
2. In Render: create a Web Service using **"Deploy an existing image from a
   registry"**, pointing at `ghcr.io/<owner>/<repo>:latest`.
3. In Render: Settings → Deploy Hook → copy the URL.
4. In GitHub: repo Settings → Secrets and variables → Actions → add
   `RENDER_DEPLOY_HOOK_URL` with that value.
