import streamlit as st


if submit:
    with st.spinner("Predicting..."):
        result = api_predict(payload)

    st.success(f"Churn Prediction: {result['churn_prediction']}")
    st.progress(result["churn_probability"])
