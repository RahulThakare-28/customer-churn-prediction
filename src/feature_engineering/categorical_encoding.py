from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DynamicColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, X_sample):
        self.ct_ = None
        self.cat_cols_ = None
        self.num_cols_ = None

    def fit(self, X, y=None):
        self.cat_cols_ = X.select_dtypes(include='object').columns.tolist()
        self.num_cols_ = X.select_dtypes(exclude='object').columns.tolist()

        self.ct_ = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.num_cols_),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.cat_cols_)
            ]
        )

        self.ct_.fit(X)
        return self

    def transform(self, X):
        return self.ct_.transform(X)



