import pandas as pd
from pathlib import Path

from src.models.xgboost_model import train_xgboost

DATA_PATH = Path("data/telco.csv")
TARGET = "Churn Label"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET])
y = df[TARGET]

metrics = train_xgboost(X, y)

print("\nXGBoost Performance")
for k, v in metrics.items():
    print(f"{k}: {v}")
