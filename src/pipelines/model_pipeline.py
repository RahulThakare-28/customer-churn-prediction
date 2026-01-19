from sklearn.pipeline import Pipeline
import joblib

def build_model_pipeline(model):
    preprocessing = joblib.load("artifacts/preprocessing.joblib")

    return Pipeline(steps=[
        ("preprocessing", preprocessing),
        ("model", model)
    ])
