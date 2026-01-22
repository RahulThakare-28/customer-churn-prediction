
import streamlit as st

st.sidebar.markdown("## 📊 Customer Churn System")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Predict Churn", "Model Insights", "Prediction History"]
)
