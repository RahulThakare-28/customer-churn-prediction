import streamlit as st
from app.ui.prediction import prediction_page
from app.ui.dashboard import dashboard_page

st.set_page_config(page_title="Customer Churn App", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "Dashboard"])

if page == "Prediction":
    prediction_page()
else:
    dashboard_page()
