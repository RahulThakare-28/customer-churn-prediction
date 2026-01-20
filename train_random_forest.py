import pandas as pd
from pathlib import Path

from src.models.random_forest import train_random_forest

DATA_PATH = Path("data/telco.csv")
TARGET = "Churn Label"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET])
y = df[TARGET]

metrics = train_random_forest(X, y)

print("\nRandom Forest Performance")
for k, v in metrics.items():
    print(f"{k}: {v}")
