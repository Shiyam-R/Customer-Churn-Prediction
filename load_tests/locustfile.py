"""
load_tests/locustfile.py
─────────────────────────────────────────────────────────────────────────────
Randomized, schema-valid traffic against /predict and /health. Weighted
5:1 toward /predict, since that's the endpoint that matters — it runs a
real SHAP TreeExplainer computation per request, not just predict_proba().

Run:
    pip install -r requirements-load.txt
    uvicorn app.main:app --host 0.0.0.0 --port 8000   # in one terminal
    locust -f load_tests/locustfile.py --host http://localhost:8000 \
        --headless -u 20 -r 5 -t 30s                   # in another
"""

import random

from locust import HttpUser, between, task

CONTRACTS = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHODS = [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)",
]
INTERNET_SERVICES = ["DSL", "Fiber optic", "No"]
YES_NO = ["Yes", "No"]


def random_customer() -> dict:
    internet = random.choice(INTERNET_SERVICES)
    has_internet = internet != "No"

    def service_value() -> str:
        return random.choice(["Yes", "No"]) if has_internet else "No internet service"

    return {
        "gender": random.choice(["Female", "Male"]),
        "SeniorCitizen": random.choice([0, 1]),
        "Partner": random.choice(YES_NO),
        "Dependents": random.choice(YES_NO),
        "tenure": random.randint(0, 72),
        "PhoneService": "Yes",
        "MultipleLines": random.choice(["Yes", "No"]),
        "InternetService": internet,
        "OnlineSecurity": service_value(),
        "OnlineBackup": service_value(),
        "DeviceProtection": service_value(),
        "TechSupport": service_value(),
        "StreamingTV": service_value(),
        "StreamingMovies": service_value(),
        "Contract": random.choice(CONTRACTS),
        "PaperlessBilling": random.choice(YES_NO),
        "PaymentMethod": random.choice(PAYMENT_METHODS),
        "MonthlyCharges": round(random.uniform(18, 120), 2),
        "TotalCharges": round(random.uniform(0, 8000), 2),
    }


class ChurnAPIUser(HttpUser):
    wait_time = between(0.5, 2)

    @task(5)
    def predict(self):
        self.client.post("/predict", json=random_customer())

    @task(1)
    def health(self):
        self.client.get("/health")
