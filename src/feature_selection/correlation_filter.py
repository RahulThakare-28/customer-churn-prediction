import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.to_drop_ = []

    def fit(self, X, y=None):
        X = X.copy()

        # ---- Only numeric columns ----
        numeric_X = X.select_dtypes(include='number')
        corr_matrix = numeric_X.corr().abs()

        # ---- Upper triangle mask ----
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # ---- Identify columns to drop ----
        self.to_drop_ = [col for col in upper.columns if any(upper[col] > self.threshold)]
        return self

    def transform(self, X):
        X = X.copy()
        # ---- Drop only the columns found in fit ----
        X = X.drop(columns=self.to_drop_, errors='ignore')
        return X
