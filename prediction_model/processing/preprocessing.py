from sklearn.base import BaseEstimator , TransformerMixin
from prediction_model.config import config
import numpy as np


class Mean_Imputer(BaseEstimator ,TranformerMixin):
    def __init__(self, variable = None):
        self.variable = variable

    def fit(self,X,y= None):
        self.mean_dict = {}
        for col in self.variable:
            self.mean_dict[col]=X[col].mean()
        return self

    def transform(self,X):
        X=X.copy()
        for col in self.variable:
            X[col].fillna(self.mean_dict[col],inplace=True)
        return X

        

class mode_Imputer(BaseEstimator ,TranformerMixin):
    def __init__(self, variable = None):
        self.variable = variable

    def fit(self,X,y= None):
        self.mode_dict = {}
        for col in self.variable:
            self.mode_dict[col]=X[col].mean()
        return self

    def transform(self,X):
        X=X.copy()
        for col in self.variable:
            X[col].fillna(self.mode_dict[col],inplace=True)
        return X



class DomainProcessing(BaseEstimator ,TranformerMixin):
    def __init__(self, variable_to_modify = None , variable_to_add= None):
        self.variable_to_modify = variable_to_modify
        self.variable_to_add = variable_to_add

    def fit(self,X,y= None):
        return self

    def transform(self,X):
        X=X.copy()
        X[variable_to_modify] = X[variable_to_modify] + X[variable_to_add]
        return X




class Drop_column(BaseEstimator ,TranformerMixin):
    def __init__(self, variable_to_drop = None):
        self.variable_to_drop = variable_to_drop


    def fit(self,X,y= None):
        return self

    def transform(self,X):
        X=X.copy()
        X = X.drop(columns=variable_to_drop)
        return X



class LabelEncoder(BaseEstimator ,TranformerMixin):
    def __init__(self, variable=None):
        self.variable = variable
        
    def fit(self,X,y= None):
        self.label_dict = {}
        for var in variable:
            t = X[var].value_counts().sort_values(ascending=True).index
            self.label_dict[var] = {k:i for i,k in enumerate(t,0)}

    def transform(self,X):
        X=X.copy()
        for feat in variable:
            X[feat] = X[feat].map(self.label_dict[feat])
        return X



class LogTransforms(BaseEstimator ,TranformerMixin):
    def __init__(self, variable=None):
        self.variable = variable
        
    def fit(self,X,y= None):
        return self

    def transform(self,X):
        X=X.copy()
        for feat in variable:
            X[feat] = np.log(X[feat])
        return X



    