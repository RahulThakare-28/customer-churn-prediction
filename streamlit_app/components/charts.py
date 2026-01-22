import streamlit as st
import pandas as pd

def churn_overview():
    data = pd.DataFrame({
        "Churn": ["Yes", "No"],
        "Count": [374, 1035]
    })

    st.subheader("Churn Distribution")
    st.bar_chart(data.set_index("Churn"))
