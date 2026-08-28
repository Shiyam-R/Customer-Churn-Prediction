"""
Quick local test for the running API — uses urllib (stdlib only, no
extra install needed). Run this with the uvicorn server already running
in another terminal.
"""

import json
import urllib.request

URL = "http://127.0.0.1:8000/predict"

payload = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0, "TotalCharges": 170.0,
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(
    URL, data=data, headers={"Content-Type": "application/json"}, method="POST"
)

with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read())
    print(json.dumps(result, indent=2))
