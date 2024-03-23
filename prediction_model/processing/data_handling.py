import os
import pandas as pd
import joblib
from prediction_model.config import config


# Load dataset
def load_dataset(file_name):
    file_path = os.path.join(config.DATAPATH, file_name)
    _data = pd.read_csv(file_path)
    return _data


# Serialization


def save_pipeline(pipeline_to_save):
    save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    joblib.dump(pipeline_to_save, save_path)
    print(f"Model has been saved under name {config.MODEL_NAME}")


# De-serialization


def load_pipeline(pipeline_to_load):
    save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    loaded_model = joblib.load(save_path)
    print("Model has been loaded")
    return loaded_model
