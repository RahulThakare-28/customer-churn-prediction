from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
from src.utils.schema import MODEL_FEATURES

# -----------------------------
# Load model (ONCE)
# -----------------------------
MODEL_PATH = Path("artifacts/selected_models/logistic_model.joblib")
model = joblib.load(MODEL_PATH)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Customer Churn Prediction API")


# -----------------------------
# Request schema (UI-friendly)
# -----------------------------
class CustomerInput(BaseModel):
    Gender: str
    Age: int
    Married: str
    Dependents: str
    Tenure_in_Months: int
    Contract: str
    Monthly_Charges: float
    Total_Charges: float
    Internet_Service: str
    Payment_Method: str


# -----------------------------
# Helper: enforce training schema
# -----------------------------
def build_input_dataframe(payload: dict) -> pd.DataFrame:
    data = {col: payload.get(col, None) for col in MODEL_FEATURES}
    return pd.DataFrame([data])


# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(data: CustomerInput):

    # Map API fields → TRAINING feature names
    payload = {
        "Gender": data.Gender,
        "Age": data.Age,
        "Married": data.Married,
        "Dependents": data.Dependents,
        "Tenure in Months": data.Tenure_in_Months,
        "Contract": data.Contract,
        "Monthly Charges": data.Monthly_Charges,
        "Total Charges": data.Total_Charges,
        "Internet Service": data.Internet_Service,
        "Payment Method": data.Payment_Method,
    }

    df = build_input_dataframe(payload)

    prob = model.predict_proba(df)[0][1]
    pred = "Yes" if prob >= 0.5 else "No"

    return {
        "churn_prediction": pred,
        "churn_probability": round(float(prob), 4),
    }


# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def health():
    return {"status": "API is running"}
