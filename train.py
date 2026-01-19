import joblib
import pandas as pd
from pathlib import Path

from src.pipelines.full_preprocessing_pipeline import build_full_pipeline

DATA_PATH = Path("data/telco.csv")

TARGET_COL = "Churn Label"

print("[INFO] Loading data...")
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

print("[INFO] Building preprocessing pipeline...")
preprocessing_pipeline = build_full_pipeline()

print("[INFO] Fitting preprocessing pipeline...")
X_processed = preprocessing_pipeline.fit_transform(X, y)

print("[INFO] Saving preprocessing pipeline...")
joblib.dump(preprocessing_pipeline, "artifacts/preprocessing.joblib")

print("[SUCCESS] Preprocessing pipeline persisted")
print("Processed shape:", X_processed.shape)
