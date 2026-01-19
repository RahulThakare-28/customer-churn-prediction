import pandas as pd
from pathlib import Path
from src.evaluation.metrics_logger import save_metrics

from src.models.logistic import train_logistic

DATA_PATH = Path("data/telco.csv")
TARGET = "Churn Label"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET])
y = df[TARGET]

pipeline, metrics = train_logistic(X, y)

print("\nLogistic Regression Performance (FIXED)")
for k, v in metrics.items():
    print(f"{k}: {v}")



baseline_metrics = {
    "accuracy": metrics["accuracy"],
    "recall": metrics["recall"],
    "roc_auc": metrics["roc_auc"]
}

save_metrics(baseline_metrics, model_name="LogisticRegression")

baseline_metrics["tn"] = metrics["confusion_matrix"][0][0]
baseline_metrics["fp"] = metrics["confusion_matrix"][0][1]
baseline_metrics["fn"] = metrics["confusion_matrix"][1][0]
baseline_metrics["tp"] = metrics["confusion_matrix"][1][1]

