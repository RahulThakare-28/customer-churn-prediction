import joblib
import pandas as pd
from pathlib import Path

from src.pipelines.preprocessing_pipeline import build_preprocessing_pipeline

DATA_PATH = Path("data/telco.csv")
TARGET = "Churn Label"
ARTIFACT_PATH = Path("src/pipelines/artifacts")

def main():
    ARTIFACT_PATH.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])

    preprocessing = build_preprocessing_pipeline()
    preprocessing.fit(X)

    joblib.dump(preprocessing, ARTIFACT_PATH / "preprocessing.joblib")
    print(" Preprocessing pipeline saved to artifacts/preprocessing.joblib")

if __name__ == "__main__":
    main()
