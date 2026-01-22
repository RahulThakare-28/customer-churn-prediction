import pandas as pd
from src.inference.schema import MODEL_FEATURES

def build_features(payload: dict) -> pd.DataFrame:
    row = {col: payload.get(col) for col in MODEL_FEATURES}
    return pd.DataFrame([row])
