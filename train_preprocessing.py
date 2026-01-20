from src.pipelines.full_preprocessing_pipeline import build_full_pipeline
import joblib
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/telco.csv")
TARGET = "Churn Label"
ARTIFACT_PATH = Path("artifacts")


def main():
    ARTIFACT_PATH.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])

    pipeline = build_full_pipeline(X)
    pipeline.fit(X)

    joblib.dump(pipeline, ARTIFACT_PATH / "preprocessing.joblib")
    print("Preprocessing pipeline saved successfully")


if __name__ == "__main__":
    main()
