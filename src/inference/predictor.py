# import pandas as pd
# from src.inference.loader import PREPROCESSING, MODEL
# from src.inference.schema import MODEL_FEATURES, API_TO_MODEL_COLS
#
#
# def build_model_input(payload: dict) -> pd.DataFrame:
#     row = {feature: None for feature in MODEL_FEATURES}
#
#     for api_field, model_field in API_TO_MODEL_COLS.items():
#         if api_field in payload:
#             row[model_field] = payload[api_field]
#
#     return pd.DataFrame([row])
#
#
# def predict(payload: dict) -> dict:
#     df = build_model_input(payload)
#
#     prob = MODEL.predict_proba(PREPROCESSING.transform(df))[0][1]
#     pred = "Yes" if prob >= 0.5 else "No"
#
#     return {
#         "churn_prediction": pred,
#         "churn_probability": round(float(prob), 4),
#     }

import joblib
from pathlib import Path
from src.inference.feature_builder import build_features

MODEL_PATH = Path("artifacts/selected_models/logistic_model.joblib")

class ChurnPredictor:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

    def predict(self, payload: dict):
        df = build_features(payload)
        prob = self.model.predict_proba(df)[0][1]
        return {
            "churn_prediction": "Yes" if prob >= 0.5 else "No",
            "churn_probability": round(prob, 4)
        }

