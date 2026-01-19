from sklearn.pipeline import Pipeline
from src.pipelines.full_preprocessing_pipeline import build_full_pipeline


def build_model_pipeline(model):
    preprocessing = build_full_pipeline()

    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessing),
        ("model", model)
    ])

    return pipeline
