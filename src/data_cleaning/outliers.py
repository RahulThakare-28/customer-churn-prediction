import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierCapper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = X.copy()
        self.bounds_ = {}

        num_cols = X.select_dtypes(include='number').columns

        for col in num_cols:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1

            # Skip constant or invalid columns
            if iqr == 0 or pd.isna(iqr):
                continue

            self.bounds_[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        return self

    def transform(self, X):
        X = X.copy()

        for col, (low, high) in self.bounds_.items():
            # 🔒 Safety guard
            if col not in X.columns:
                continue
            if not pd.api.types.is_numeric_dtype(X[col]):
                continue

            X[col] = X[col].clip(lower=low, upper=high)

        return X
