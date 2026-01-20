import pandas as pd
from app.backend.model_loader import model


def predict_churn(data: dict):
    df = pd.DataFrame([data])

    prob = model.predict_proba(df)[0][1]
    pred = "Yes" if prob >= 0.5 else "No"

    return {
        "churn_prediction": pred,
        "churn_probability": round(prob, 4)
    }
