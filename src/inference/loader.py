import joblib
from pathlib import Path

PREPROCESSING_PATH = Path("artifacts/preprocessing.joblib")
MODEL_PATH = Path("artifacts/selected_models/best_model.joblib")  # Best selected

def load_artifacts():
    preprocessing = joblib.load(PREPROCESSING_PATH)
    model = joblib.load(MODEL_PATH)
    return preprocessing, model

PREPROCESSING, MODEL = load_artifacts()
