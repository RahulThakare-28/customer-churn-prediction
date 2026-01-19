import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureCreator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        debug = False  # set False to disable printing

        if debug:
            print("[DEBUG] Total Revenue sample:", X['Total Revenue'].head(20).tolist())
            print("[DEBUG] Tenure in Months sample:", X['Tenure in Months'].head(20).tolist())

        # ---- SAFE numeric conversion ----
        if 'Total Revenue' in X.columns:
            X['Total Revenue'] = pd.to_numeric(
                X['Total Revenue'], errors='coerce'
            )

        if 'Tenure in Months' in X.columns:
            X['Tenure in Months'] = pd.to_numeric(
                X['Tenure in Months'], errors='coerce'
            )

        # ---- Feature 1: AvgRevenuePerMonth ----
        if {'Total Revenue', 'Tenure in Months'}.issubset(X.columns):
            X['AvgRevenuePerMonth'] = (
                X['Total Revenue'] / (X['Tenure in Months'] + 1)
            )
        else:
            X['AvgRevenuePerMonth'] = 0

        # ---- Feature 2: ServiceCount ----
        service_cols = [
            'Streaming TV',
            'Streaming Movies',
            'Streaming Music'
        ]

        existing_services = [c for c in service_cols if c in X.columns]

        X['ServiceCount'] = 0
        for col in existing_services:
            X['ServiceCount'] += (X[col] == 'Yes').astype(int)

        return X

