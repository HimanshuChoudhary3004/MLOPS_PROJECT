import numpy as np
from sklearn.pipeline import Pipeline
from prediction_model.config import config
import prediction_model.processing.preprocessing as pp
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression


classification_pipeline = Pipeline(
    [
        ("meanimputer", pp.Mean_Imputer(config.NUM_FEATURES)),
        ("modeimputer", pp.Mode_Imputer(config.CAT_FEATURES)),
        ("domainprocessing",pp.DomainProcessing(variable_to_modify=config.FEATURE_TO_MODIFY,variable_to_add=config.FEATURE_TO_ADD)),
        ("dropcolumn", pp.Drop_column(variable_to_drop=config.DROP_FEATURE)),
        ("labelencoder", pp.LabelEncoder(config.FEATURES_TO_ENCODE)),
        ("logtransformation", pp.LogTransforms(config.LOG_FEATURES)),
        ("scaling", MinMaxScaler()),
        ("LogisticClassifier", LogisticRegression(random_state=0)),
    ]
)
