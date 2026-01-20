import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("📊 Customer Churn Prediction")

with st.form("churn_form"):
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Age = st.number_input("Age", min_value=18, max_value=100, step=1)

    Married = st.selectbox("Married", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])

    Tenure_in_Months = st.number_input(
        "Tenure in Months", min_value=0, max_value=100, step=1
    )

    Contract = st.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )

    Monthly_Charges = st.number_input(
        "Monthly Charges", min_value=0.0, step=1.0
    )

    Total_Charges = st.number_input(
        "Total Charges", min_value=0.0, step=10.0
    )

    Internet_Service = st.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )

    Payment_Method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

    submitted = st.form_submit_button("Predict")


if submitted:
    payload = {
        "Gender": Gender,
        "Age": Age,
        "Married": Married,
        "Dependents": Dependents,
        "Tenure_in_Months": Tenure_in_Months,
        "Contract": Contract,
        "Monthly_Charges": Monthly_Charges,
        "Total_Charges": Total_Charges,
        "Internet_Service": Internet_Service,
        "Payment_Method": Payment_Method,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=5)

        if response.status_code == 200:
            result = response.json()

            st.success(f"Churn Prediction: **{result['churn_prediction']}**")
            st.info(f"Churn Probability: **{result['churn_probability']}**")

        else:
            st.error(f"API Error: {response.text}")

    except requests.exceptions.RequestException:
        st.error("❌ FastAPI server not reachable. Is it running?")
