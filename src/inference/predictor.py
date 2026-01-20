import pandas as pd
from src.inference.loader import PREPROCESSING, MODEL
from src.inference.schema import MODEL_FEATURES, API_TO_MODEL_COLS

def predict_from_api(payload: dict) -> dict:
    """
    payload: API JSON dictionary
    Returns: prediction and probability
    """

    # 1️⃣ Initialize all features with None
    row = {feat: None for feat in MODEL_FEATURES}

    # 2️⃣ Map API fields → MODEL_FEATURES
    for api_field, model_field in API_TO_MODEL_COLS.items():
        if api_field in payload:
            row[model_field] = payload[api_field]

    # 3️⃣ Create DataFrame
    df = pd.DataFrame([row])

    # 4️⃣ Preprocess & predict
    prob = MODEL.predict_proba(PREPROCESSING.transform(df))[0][1]
    pred = "Yes" if prob >= 0.5 else "No"

    return {"churn_prediction": pred, "churn_probability": round(prob, 4)}
