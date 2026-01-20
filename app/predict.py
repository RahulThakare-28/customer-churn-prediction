import joblib
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

model = joblib.load(
    ROOT / "artifacts/selected_models/xgboost_model.joblib"
)

def predict_churn(customer_data: dict):
    df = pd.DataFrame([customer_data])

    # Pipeline handles preprocessing internally
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "churn": int(prediction),
        "probability": float(probability)
    }

sample_customer = {
    "Gender": "Male",
    "Senior Citizen": "No",
    "Partner": "Yes",

}

result = predict_churn(sample_customer)
print(result)