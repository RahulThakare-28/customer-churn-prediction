import streamlit as st
import pandas as pd

def churn_distribution_chart(df: pd.DataFrame):
    churn_counts = df["churn"].value_counts()
    st.bar_chart(churn_counts)

def churn_trend_chart(df: pd.DataFrame):
    trend = df.groupby(df["timestamp"].dt.date)["churn"].mean()
    st.line_chart(trend)
