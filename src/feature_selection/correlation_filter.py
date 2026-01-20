import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.to_drop_ = []
        self.numeric_cols_ = []

    def fit(self, X, y=None):
        # Ensure DataFrame
        X = pd.DataFrame(X).copy()

        # Select numeric columns only
        numeric_X = X.select_dtypes(include=[np.number])
        self.numeric_cols_ = numeric_X.columns.tolist()

        # Edge case: no numeric columns
        if numeric_X.shape[1] <= 1:
            self.to_drop_ = []
            return self

        corr_matrix = numeric_X.corr().abs()

        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        self.to_drop_ = [
            col for col in upper.columns
            if any(upper[col] > self.threshold)
        ]

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        # Drop only columns found during fit
        if self.to_drop_:
            X = X.drop(columns=self.to_drop_, errors="ignore")

        return X
