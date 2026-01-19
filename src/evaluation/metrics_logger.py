import json
import pandas as pd
from pathlib import Path


METRICS_DIR = Path("artifacts/metrics")
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def save_metrics(metrics: dict, model_name: str):
    """
    Save metrics in both CSV (for comparison)
    and JSON (for detailed inspection).
    """

    # ---- JSON (single model) ----
    json_path = METRICS_DIR / f"{model_name}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # ---- CSV (append or create) ----
    csv_path = METRICS_DIR / "baseline_metrics.csv"

    row = metrics.copy()
    row["model"] = model_name

    df_row = pd.DataFrame([row])

    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        df_all = pd.concat([df_existing, df_row], ignore_index=True)
    else:
        df_all = df_row

    df_all.to_csv(csv_path, index=False)
