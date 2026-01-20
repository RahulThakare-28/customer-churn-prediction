from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

# -----------------------------
# Load model (ONCE)
# -----------------------------
MODEL_PATH = Path("artifacts/selected_models/logistic_model.joblib")
model = joblib.load(MODEL_PATH)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Customer Churn Prediction API")


class CustomerInput(BaseModel):
    Gender: str
    Age: int
    Married: str
    Dependents: str
    Tenure_in_Months: int
    Contract: str
    Monthly_Charge: float
    Total_Revenue: float
    Internet_Service: str
    Payment_Method: str


@app.post("/predict")
def predict(data: CustomerInput):

    #  MAP API FIELDS → TRAINING FEATURE NAMES
    row = {
        "Gender": data.Gender,
        "Age": data.Age,
        "Married": data.Married,
        "Dependents": data.Dependents,
        "Tenure in Months": data.Tenure_in_Months,
        "Contract": data.Contract,
        "Monthly Charge": data.Monthly_Charge,
        "Total Revenue": data.Total_Revenue,
        "Internet Service": data.Internet_Service,
        "Payment Method": data.Payment_Method,
    }

    df = pd.DataFrame([row])

    prob = model.predict_proba(df)[0][1]
    pred = "Yes" if prob >= 0.5 else "No"

    return {
        "churn_prediction": pred,
        "churn_probability": round(prob, 4)
    }

@app.get("/")
def root():
    return {"message": "API is running"}

@app.get("/")
def health():
    return {"status": "ok"}


