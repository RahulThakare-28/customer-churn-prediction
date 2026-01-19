from sklearn.base import BaseEstimator, TransformerMixin

class LeakageRemover(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.leakage_cols = [
            "Customer Status",
            "Churn Score",
            "Churn Category",
            "Churn Reason"
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.leakage_cols, errors="ignore")
