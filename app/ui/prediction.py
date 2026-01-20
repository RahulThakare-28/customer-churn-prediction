import streamlit as st
from datetime import datetime
from app.services.model_service import ChurnModelService
from app.services.mongo_service import MongoService

def prediction_page():
    st.header(" Customer Churn Prediction")

    model_service = ChurnModelService()
    mongo_service = MongoService()

    with st.form("prediction_form"):
        customer_id = st.text_input("Customer ID")
        customer_segment = st.selectbox("Customer Segment", ["Retail", "SME", "Enterprise"])
        tenure = st.number_input("Tenure (months)", 0, 100)
        monthly_charges = st.number_input("Monthly Charges", 0.0)
        total_charges = st.number_input("Total Charges", 0.0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = {
            "customer_segment": customer_segment,
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges
        }

        churn, prob = model_service.predict(input_data)

        st.success(f"Churn: {churn} | Probability: {prob:.2f}")

        mongo_service.insert_prediction({
            **input_data,
            "customer_id": customer_id,
            "churn": churn,
            "probability": prob,
            "timestamp": datetime.utcnow()
        })
