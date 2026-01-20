from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

PREPROCESSING_PATH = Path("artifacts/preprocessing.joblib")


def build_model_pipeline(model):
    if not PREPROCESSING_PATH.exists():
        raise FileNotFoundError(
            "Preprocessing pipeline not found. "
            "Run src/pipelines/train_preprocessing.py first."
        )

    preprocessing = joblib.load(PREPROCESSING_PATH)

    return Pipeline(steps=[
        ("preprocessing", preprocessing),
        ("model", model)
    ])
