import pandas as pd
from src.inference.loader import PREPROCESSING, MODEL
from src.inference.schema import MODEL_FEATURES, API_TO_MODEL_COLS


def build_model_input(payload: dict) -> pd.DataFrame:
    row = {feature: None for feature in MODEL_FEATURES}

    for api_field, model_field in API_TO_MODEL_COLS.items():
        if api_field in payload:
            row[model_field] = payload[api_field]

    return pd.DataFrame([row])


def predict(payload: dict) -> dict:
    df = build_model_input(payload)

    prob = MODEL.predict_proba(PREPROCESSING.transform(df))[0][1]
    pred = "Yes" if prob >= 0.5 else "No"

    return {
        "churn_prediction": pred,
        "churn_probability": round(float(prob), 4),
    }
