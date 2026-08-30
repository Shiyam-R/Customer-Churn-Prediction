"""
dashboard/app.py
─────────────────────────────────────────────────────────────────────────────
Minimal Streamlit dashboard for the Customer Churn Prediction API.

Fixes for Render deployment connectivity:
- Normalizes pasted URLs (removes /docs, /health, /predict, trailing slashes).
- Uses a persistent requests.Session with retries for transient Render failures.
- Gives the deployed service longer to respond, especially after a cold start.
- Checks /health independently from optional /version metadata.
- Shows the actual connection error instead of only a generic message.
- Uses HTTP status validation before parsing JSON.
"""

import time

import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DEFAULT_API_URL = "https://customer-churn-prediction-xrp5.onrender.com"

# Render/free-tier services can occasionally take longer to wake up.
HEALTH_TIMEOUT = (10, 30)   # (connect timeout, read timeout)
REQUEST_TIMEOUT = (10, 45)  # (connect timeout, read timeout)


def normalize_api_url(url: str) -> str:
    """Convert a pasted Render API/docs endpoint into the API base URL."""
    url = url.strip().rstrip("/")

    # Users often paste a Swagger/docs or endpoint URL into the API URL field.
    for suffix in ("/docs", "/openapi.json", "/health", "/version", "/predict", "/drift"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]

    return url.rstrip("/")


@st.cache_resource
def get_http_session() -> requests.Session:
    """Create one reusable HTTP session with retries for transient failures."""
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)

    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_json(session: requests.Session, url: str, timeout):
    """
    Make a GET request and safely return JSON.

    Raises RequestException for HTTP/network failures and ValueError for
    unexpected non-JSON responses.
    """
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
)

session = get_http_session()

# ── Sidebar: API connection + live status ────────────────────────────────────
with st.sidebar:
    st.header("API Connection")

    raw_api_url = st.text_input(
        "API URL",
        value=DEFAULT_API_URL,
        help="Paste the API base URL. A /docs URL is also accepted and will be normalized.",
    )
    api_url = normalize_api_url(raw_api_url)

    if not api_url.startswith(("http://", "https://")):
        st.error("API URL must start with http:// or https://")
        api_reachable = False
    else:
        api_reachable = False

        with st.spinner("Checking API connection..."):
            try:
                health = get_json(session, f"{api_url}/health", HEALTH_TIMEOUT)

                # Reaching /health successfully is enough to consider the API reachable.
                api_reachable = True

                if health.get("model_loaded"):
                    st.success("Connected — model loaded")
                else:
                    st.warning("Connected — model NOT loaded")

            except requests.exceptions.Timeout:
                st.error(
                    "API connection timed out. Render may be waking up from a cold start. "
                    "Wait a moment and refresh."
                )
            except requests.exceptions.RequestException as e:
                st.error("Could not reach API")
                st.caption(f"Connection details: {e}")
            except ValueError:
                st.error("API responded, but /health did not return valid JSON.")

        # Version metadata is useful but should NOT mark the API as unreachable.
        if api_reachable:
            try:
                version = get_json(session, f"{api_url}/version", HEALTH_TIMEOUT)
                st.caption(
                    f"Model version: {version.get('model_version', 'unknown')}"
                )
                st.caption(
                    f"Trained: {str(version.get('model_trained_at', 'unknown'))[:19]}"
                )
                st.caption(
                    f"Environment: {version.get('environment', 'unknown')}"
                )
            except (requests.exceptions.RequestException, ValueError):
                st.caption("Model version metadata unavailable.")

    with st.expander("Feature drift (live traffic vs. training)"):
        if not api_reachable:
            st.caption("Connect to the API to load the drift report.")
        else:
            try:
                drift = get_json(session, f"{api_url}/drift", HEALTH_TIMEOUT)

                if drift.get("status") == "insufficient_data":
                    st.caption(
                        f"Only {drift.get('live_sample_size', 0)}/"
                        f"{drift.get('min_samples_required', 'unknown')} "
                        "live requests so far — not enough to report drift yet."
                    )
                else:
                    st.caption(
                        f"Based on {drift.get('live_sample_size', 'unknown')} "
                        "recent live requests."
                    )

                    for feat in drift.get("features", [])[:5]:
                        st.text(
                            f"{feat.get('feature', 'unknown')}: "
                            f"PSI {feat.get('psi', 'unknown')} "
                            f"({feat.get('severity', 'unknown')})"
                        )

            except requests.exceptions.Timeout:
                st.caption("Drift report timed out.")
            except (requests.exceptions.RequestException, ValueError):
                st.caption("Drift report unavailable.")


# ── Main page ────────────────────────────────────────────────────────────────
st.title("📉 Customer Churn Prediction")
st.caption("Enter a customer's account and service details to predict churn risk.")

# ── Prediction form ─────────────────────────────────────────────────────────
with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Account")

        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x else "No",
        )
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=12,
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
        )
        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"],
        )
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    with col2:
        st.subheader("Services")

        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"],
        )
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
        )
        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"],
        )
        online_backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"],
        )
        device_protection = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"],
        )
        tech_support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"],
        )
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"],
        )
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"],
        )

    st.subheader("Billing")
    col3, col4 = st.columns(2)

    with col3:
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=500.0,
            value=70.0,
            step=0.5,
        )

    with col4:
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=20000.0,
            value=1000.0,
            step=10.0,
        )

    submitted = st.form_submit_button(
        "Predict Churn",
        type="primary",
        use_container_width=True,
    )


# ── Handle submission ─────────────────────────────────────────────────────────
if submitted:
    if not api_reachable:
        st.error(
            "Can't predict because the API health check failed. "
            "Check the URL and try again after the Render service is awake."
        )

    else:
        payload = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        with st.spinner("Calling the model..."):
            try:
                response = session.post(
                    f"{api_url}/predict",
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )
            except requests.exceptions.Timeout:
                st.error(
                    "Prediction timed out. The deployed service may be under load "
                    "or waking up."
                )
                response = None
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")
                response = None

        if response is not None:
            if response.status_code == 200:
                try:
                    result = response.json()
                except ValueError:
                    st.error("The API returned a successful response, but it was not valid JSON.")
                    result = None

                if result is not None:
                    proba = result["churn_probability"]
                    threshold = result["threshold_used"]

                    st.divider()
                    res_col1, res_col2 = st.columns([1, 2])

                    with res_col1:
                        if result["prediction"] == "Churn":
                            st.error(f"⚠️ **{result['prediction']}**")
                        else:
                            st.success(f"✅ **{result['prediction']}**")

                        st.metric("Churn Probability", f"{proba:.1%}")
                        st.caption(
                            f"Decision threshold: {threshold:.1%} "
                            "(cost-sensitive, not the default 50%)"
                        )

                    with res_col2:
                        st.subheader("Top Contributing Factors (this customer)")

                        factors_df = pd.DataFrame(
                            result["top_contributing_factors"]
                        )

                        if not factors_df.empty:
                            factors_df["direction"] = factors_df["shap_value"].apply(
                                lambda value: (
                                    "Pushes toward churn"
                                    if value > 0
                                    else "Pushes away from churn"
                                )
                            )

                            st.bar_chart(
                                factors_df.set_index("feature")["shap_value"]
                            )
                            st.dataframe(
                                factors_df,
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            st.info("No contributing factors were returned by the API.")

            elif response.status_code == 429:
                st.warning(
                    "Rate limit exceeded — please wait a moment and try again."
                )

            elif response.status_code == 422:
                st.warning("The API rejected this input as invalid:")
                try:
                    st.json(response.json())
                except ValueError:
                    st.code(response.text)

            else:
                st.error(
                    f"API error ({response.status_code}): {response.text}"
                )
