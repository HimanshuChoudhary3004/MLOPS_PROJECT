import pandas as pd
import numpy as np
from prediction_model.config import config
from prediction_model.preprocessing.data_handling import load_dataset , save_pipeline
from prediction_model.preprocessing.preprocessing as pp
from prediction_model.pipeline as pipe


def perform_training():
    train_data = load_dataset(config.TRAIN_FILE)
    train_target = train_data(config.TARGET).map({'N':0 , 'Y':1})
    pipe.classification_pipeline.fit(train_data[config.FEATURES] , train_target)
    save_pipeline(pipe.classification_pipeline)

if __name__='__main__':
    perform_training()


