import streamlit as st
import pandas as pd
from app.services.mongo_service import MongoService
from app.utils.charts import churn_trend_chart, churn_distribution_chart

def dashboard_page():
    st.header(" Churn Analytics Dashboard")

    mongo_service = MongoService()
    df = mongo_service.fetch_all()

    if df.empty:
        st.warning("No prediction data available.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Sidebar filters
    st.sidebar.header("Filters")
    segment = st.sidebar.multiselect(
        "Customer Segment",
        options=df["customer_segment"].unique(),
        default=df["customer_segment"].unique()
    )

    date_range = st.sidebar.date_input(
        "Date Range",
        [df["timestamp"].min().date(), df["timestamp"].max().date()]
    )

    filtered_df = df[
        (df["customer_segment"].isin(segment)) &
        (df["timestamp"].dt.date >= date_range[0]) &
        (df["timestamp"].dt.date <= date_range[1])
    ]

    st.subheader("Churn Distribution")
    churn_distribution_chart(filtered_df)

    st.subheader("Historical Churn Trend")
    churn_trend_chart(filtered_df)

    st.subheader("Download Data")
    st.download_button(
        "Download CSV",
        filtered_df.to_csv(index=False),
        file_name="churn_predictions.csv"
    )
