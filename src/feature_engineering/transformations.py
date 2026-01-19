import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols


    def fit(self, X, y=None):
        return self


    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col] = np.log1p(X[col])
        return X