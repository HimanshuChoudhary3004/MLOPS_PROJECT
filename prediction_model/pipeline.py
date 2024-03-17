import numpy as np
from sklearn.pipeline import pipeline
from prediction_model.config import config
from prediction_model.preprocessing.preprocessing as pp
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression


classification_pipeline = pipeline(
            [
                ('meanimputer',pp.Mean_Imputer(config.NUM_FEATURES)),
                ('modeimputer',pp.mode_Imputer(config.CAT_FEATURES)),
                ('domainprocessing',pp.DomainProcessing(variable_to_modify=config.FEATURE_TO_MODIFY,variable_to_add=config.FEATURE_TO_ADD))
                ('dropcolumn',pp.Drop_column(variable_to_drop=config.DROP_FEATURE))
                ('labelencoder',pp.LabelEncoder(config.FEATURES_TO_ENCODE))
                ('logtransformation',pp.LogTransforms(config.LOG_FEATURES))
                ('scaling',MinMaxScaler())
                ('LogisticClassifier'LogisticRegression(random_state=0))
                
            ]
    )