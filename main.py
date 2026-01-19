

import pandas as pd
from pathlib import Path

from src.pipelines.full_preprocessing_pipeline import build_full_pipeline


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
DATA_PATH = Path("data/telco.csv")
TARGET_COL = "Churn Label"


# ------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------

def main():
    print("[INFO] Loading raw dataset...")
    df = pd.read_csv(DATA_PATH)

    print(f"[INFO] Dataset shape: {df.shape}")

    # -----------------------------
    # Separate features & target
    # -----------------------------
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    print(f"[INFO] Features shape before preprocessing: {X.shape}")

    # -----------------------------
    # Build & run pipeline
    # -----------------------------
    print("[INFO] Building preprocessing pipeline...")
    pipeline = build_full_pipeline()

    print("[INFO] Fitting & transforming data...")
    X_processed = pipeline.fit_transform(X, y)

    # -----------------------------
    # Sanity checks
    # -----------------------------
    print("[SUCCESS] Pipeline executed successfully")
    print(f"[INFO] Processed feature matrix shape: {X_processed.shape}")
    print(f"[INFO] Target distribution:\n{y.value_counts()}")


# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------

if __name__ == "__main__":
    main()
