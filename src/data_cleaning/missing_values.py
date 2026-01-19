# import pandas as pd
#
# from sklearn.base import BaseEstimator, TransformerMixin
#
#
# class MissingValueHandler(BaseEstimator, TransformerMixin):
#     def __init__(self, threshold=0.5):
#         self.threshold = threshold
#         self.num_cols = None
#         self.cat_cols = None
#
#     def fit(self, X, y=None):
#         missing_ratio = X.isnull().mean()
#         self.drop_cols_ = missing_ratio[missing_ratio > self.threshold].index.tolist()
#         self.num_cols = X.select_dtypes(exclude='object').columns
#         self.cat_cols = X.select_dtypes(include='object').columns
#         return self
#
#
#     def transform(self, X):
#         X = X.drop(columns=self.drop_cols_, errors='ignore').copy()
#         for col in self.num_cols:
#             X[col].fillna(X[col].median(), inplace=True)
#
#         for col in self.cat_cols:
#             X[col].fillna(X[col].mode()[0], inplace=True)
#         return X


import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MissingValueHandler(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.5, protected_cols=None):
        self.threshold = threshold
        self.protected_cols = protected_cols or []
        self.drop_cols_ = None
        self.num_cols_ = None
        self.cat_cols_ = None

    def fit(self, X, y=None):
        X = X.copy()

        # Missing ratio
        missing_ratio = X.isnull().mean()

        # Columns to drop (except protected ones)
        self.drop_cols_ = [
            col for col in missing_ratio[missing_ratio > self.threshold].index
            if col not in self.protected_cols
        ]

        # Remaining columns AFTER drop
        remaining_cols = X.drop(columns=self.drop_cols_, errors="ignore")

        self.num_cols_ = remaining_cols.select_dtypes(exclude="object").columns.tolist()
        self.cat_cols_ = remaining_cols.select_dtypes(include="object").columns.tolist()

        return self

    def transform(self, X):
        X = X.copy()

        # Drop columns
        X = X.drop(columns=self.drop_cols_, errors="ignore")

        # Fill numerical columns
        for col in self.num_cols_:
            if col in X.columns:
                X[col] = X[col].fillna(X[col].median())

        # Fill categorical columns
        for col in self.cat_cols_:
            if col in X.columns:
                X[col] = X[col].fillna("No Offer")

        return X
