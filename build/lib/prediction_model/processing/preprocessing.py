from sklearn.base import BaseEstimator, TransformerMixin
from prediction_model.config import config
import numpy as np


class Mean_Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, variable=None):
        self.variable = variable

    def fit(self, X, y=None):
        self.mean_dict = {}
        for col in self.variable:
            self.mean_dict[col] = X[col].mean()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.variable:
            X[col] = X[col].fillna(self.mean_dict[col])
        return X


class Mode_Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, variable=None):
        self.variable = variable

    def fit(self, X, y=None):
        self.mode_dict = {}
        for col in self.variable:
            self.mode_dict[col] = X[col].mode().iloc[0]  # Get the mode value
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.variable:
            X[col] = X[col].fillna(self.mode_dict[col])  # Fill NaN values with mode
        return X


class DomainProcessing(BaseEstimator, TransformerMixin):
    def __init__(self, variable_to_modify=None, variable_to_add=None):
        self.variable_to_modify = variable_to_modify
        self.variable_to_add = variable_to_add

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variable_to_modify:
            X[feature] = X[feature] + X[self.variable_to_add]
        return X


class Drop_column(BaseEstimator, TransformerMixin):
    def __init__(self, variable_to_drop=None):
        self.variable_to_drop = variable_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(columns=self.variable_to_drop)
        return X


class LabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variable=None):
        self.variable = variable
        self.label_dict = {}

    def fit(self, X, y=None):
        for var in self.variable:
            t = X[var].value_counts().sort_values(ascending=True).index
            self.label_dict[var] = {k: i for i, k in enumerate(t, 0)}
        return self

    def transform(self, X):
        X = X.copy()
        for feat in self.variable:
            if feat in self.label_dict:
                X[feat] = X[feat].map(self.label_dict[feat])
        return X


class LogTransforms(BaseEstimator, TransformerMixin):
    def __init__(self, variable=None):
        self.variable = variable

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feat in self.variable:
            X[feat] = np.log(X[feat])
        return X
