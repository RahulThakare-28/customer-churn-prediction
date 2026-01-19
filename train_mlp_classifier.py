import pandas as pd
from pathlib import Path

from src.models.mlp_classifier import train_mlp
from src.models.mlp_cv import evaluate_mlp_cv

DATA_PATH = Path("data/telco.csv")
TARGET = "Churn Label"


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # ---- Train / Test Evaluation ----
    metrics = train_mlp(X, y)

    print("\nMLPClassifier Performance (Hold-out Test Set)")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # ---- Cross-Validation Evaluation ----
    cv_metrics = evaluate_mlp_cv(X, y)

    print("\nMLPClassifier Cross-Validation Performance")
    for k, v in cv_metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
