from sklearn.base import BaseEstimator, TransformerMixin


class TypeCaster(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self


    def transform(self, X):
        X = X.copy()
        for col in X.select_dtypes(include='object').columns:
            X[col] = X[col].astype(str)
        return X