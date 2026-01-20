import joblib
from pathlib import Path

ARTIFACTS = Path("artifacts")
MODEL_PATH = ARTIFACTS / "selected_models/logistic_model.joblib"

model = joblib.load(MODEL_PATH)
