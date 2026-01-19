import pandas as pd
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


from src.models.logistic import train_logistic
from src.models.random_forest import train_random_forest
from src.models.adaboost import train_adaboost
from src.models.xgboost_model import train_xgboost
from src.models.mlp_classifier import train_mlp

MODELS = {
    "LogisticRegression": train_logistic,
    "RandomForest": train_random_forest,
    "AdaBoost": train_adaboost,
    "XGBoost": train_xgboost,
    "MLPClassifier": train_mlp,
}

DATA_PATH = Path("data/telco.csv")
TARGET = "Churn Label"


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    results = []

    print("\n===== MODEL TRAINING STARTED =====")

    for name, train_fn in MODELS.items():
        print(f"Training {name} ...", end=" ")

        metrics = dict(train_fn(X, y))
        metrics["model"] = name
        results.append(metrics)

        print("done")

    results_df = (
        pd.DataFrame(results)
        .sort_values("roc_auc", ascending=False)
        .reset_index(drop=True)
    )

    print("\n===== FINAL MODEL COMPARISON =====")
    print(results_df[["model", "accuracy", "recall", "roc_auc"]])

    best_model = results_df.loc[0, "model"]
    print(f"\n🏆 Best Model Selected: {best_model}")


if __name__ == "__main__":
    main()
