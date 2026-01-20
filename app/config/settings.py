from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "artifacts/selected_models/xgboost_model.joblib"
PREPROCESSING_PATH = BASE_DIR / "artifacts/preprocessing.joblib"

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "churn_db"
COLLECTION_NAME = "predictions"
