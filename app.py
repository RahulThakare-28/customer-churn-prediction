import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("📊 Customer Churn Prediction")

with st.form("churn_form"):
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Age = st.number_input("Age", 18, 100)
    Married = st.selectbox("Married", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    Tenure_in_Months = st.number_input("Tenure in Months", 0, 100)
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    Monthly_Charge = st.number_input("Monthly Charge")
    Total_Revenue = st.number_input("Total Revenue")
    Internet_Service = st.selectbox("Internet Service", ["Yes", "No"])
    Payment_Method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
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
        "Monthly_Charge": Monthly_Charge,
        "Total_Revenue": Total_Revenue,
        "Internet_Service": Internet_Service,
        "Payment_Method": Payment_Method,
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()
        st.success(f"Churn Prediction: {result['churn_prediction']}")
        st.info(f"Churn Probability: {result['churn_probability']}")
    else:
        st.error("Prediction failed. Is FastAPI running?")
