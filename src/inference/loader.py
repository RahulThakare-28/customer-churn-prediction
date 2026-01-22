import joblib
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")

PREPROCESSING_PATH = ARTIFACTS_DIR / "preprocessing.joblib"
MODEL_PATH = ARTIFACTS_DIR / "selected_models" / "logistic_model.joblib"

def load_artifacts():
    preprocessing = joblib.load(PREPROCESSING_PATH)
    model = joblib.load(MODEL_PATH)
    return preprocessing, model

PREPROCESSING, MODEL = load_artifacts()
