import requests

API_URL = "http://127.0.0.1:8000/predict"

def predict_churn(payload):
    r = requests.post(API_URL, json=payload)
    r.raise_for_status()
    return r.json()
