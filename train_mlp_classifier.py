import pandas as pd
from pathlib import Path

from src.models.mlp_classifier import train_mlp

DATA_PATH = Path("data/telco.csv")
TARGET = "Churn Label"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET])
y = df[TARGET]

metrics = train_mlp(X, y)

print("\nMLPClassifier Performance")
for k, v in metrics.items():
    print(f"{k}: {v}")
