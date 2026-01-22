import streamlit as st

st.title("🔮 Predict Customer Churn")

with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    with col2:
        monthly = st.number_input("Monthly Charge", 0.0, 200.0, 70.0)
        total = st.number_input("Total Charges", 0.0, 10000.0)
        payment = st.selectbox("Payment Method", [...])

    submit = st.form_submit_button("Predict")
