import json
import pandas as pd
from pathlib import Path
import numpy as np

METRICS_DIR = Path("artifacts/metrics")
METRICS_DIR.mkdir(parents=True, exist_ok=True)

def _to_python_type(obj):
    """
    Convert numpy types to native Python types
    for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def save_metrics(metrics: dict, model_name: str):
    safe_metrics = {k: _to_python_type(v) for k, v in metrics.items()}

    # ---- JSON ----
    json_path = METRICS_DIR / f"{model_name}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(safe_metrics, f, indent=4)

    # ---- CSV ----
    row = safe_metrics.copy()
    row["model"] = model_name

    df_row = pd.DataFrame([row])

    csv_path = METRICS_DIR / "baseline_metrics.csv"

    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        df_all = pd.concat([df_existing, df_row], ignore_index=True)
    else:
        df_all = df_row

    df_all.to_csv(csv_path, index=False)
