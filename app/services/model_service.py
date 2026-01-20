

import joblib
import pandas as pd
from app.config.settings import MODEL_PATH, PREPROCESSING_PATH

class ChurnModelService:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.preprocessor = joblib.load(PREPROCESSING_PATH)

    def predict(self, input_data: dict):
        df = pd.DataFrame([input_data])
        X = self.preprocessor.transform(df)

        prediction = int(self.model.predict(X)[0])
        probability = float(self.model.predict_proba(X)[0][1])

        return prediction, probability

    def feature_importance(self):
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return None
