import streamlit as st
from pymongo import MongoClient
from app.predict import predict_churn  # your function from above

st.title("Customer Churn Prediction")

# Input form
with st.form("customer_form"):
    customer_id = st.text_input("Customer ID")
    feature1 = st.number_input("Feature 1")
    feature2 = st.number_input("Feature 2")
    # ... add all necessary features
    submitted = st.form_submit_button("Predict")

if submitted:
    data = {
        "customer_id": customer_id,
        "feature1": feature1,
        "feature2": feature2,
        # ... add all features
    }
    result = predict_churn(data)
    st.success(f"Predicted Churn: {result['churn']}, Probability: {result['probability']:.2f}")

    # Save to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["churn_db"]
    db["predictions"].insert_one({**data, **result})
