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
request.

Layered structure: `app/main.py` (app factory + lifespan startup) →
`app/api/routes.py` → `app/services/prediction_service.py` →
`app/model_loader.py` (artifacts loaded once at startup) +
`app/utils/feature_engineering.py` (inference-only, no training dependency).

## Docker

```
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## Reports

`reports/` — EDA charts, PR curve, confusion matrices, SHAP summary from
the eval pass.
