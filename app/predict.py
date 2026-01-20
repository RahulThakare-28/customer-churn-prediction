import joblib
import pandas as pd

preprocessing = joblib.load("artifacts/preprocessing.joblib")
model = joblib.load("artifacts/selected_models/xgboost_model.joblib")

def predict_churn(customer_data: dict):
    # Convert dict to DataFrame
    df = pd.DataFrame([customer_data])
    # Preprocess
    X_processed = preprocessing.transform(df)
    # Predict
    prediction = model.predict(X_processed)[0]
    probability = model.predict_proba(X_processed)[0][1]
    return {"churn": int(prediction), "probability": float(probability)}
