from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
from src.feature_selection.importance_selector import importance_selector
from src.feature_selection.pca_extractor import pca_transformer

PREPROCESSING_PATH = Path("artifacts/preprocessing.joblib")


def build_model_pipeline(model):
    if not PREPROCESSING_PATH.exists():
        raise FileNotFoundError(
            "Preprocessing pipeline not found. "
            "Run train_preprocessing.py first."
        )

    preprocessing = joblib.load(PREPROCESSING_PATH)

    return Pipeline(steps=[
        ("preprocessing", preprocessing),
        ("feature_select", importance_selector()),
        ("pca", pca_transformer()),
        ("model", model)
])

