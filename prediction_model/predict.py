import pandas as pd
import numpy as np
import joblib
from prediction_model.config import config
from prediction_model.processing.data_handling import load_pipeline
from prediction_model.processing.data_handling import load_dataset
import prediction_model.pipeline as pipe


classification_pipeline = load_pipeline(config.MODEL_NAME)


def generate_prediction(data_input):
    data = pd.DataFrame(data_input)
    pred = classification_pipeline.predict(data[config.FEATURES])
    output = np.where(pred == 1, "Y", "N")
    result = {"Prediction": output}
    return result


# def generate_prediction():
#   test_data = load_dataset(config.TEST_FILE)
# output = np.where(pred==1,'Y','N')
# print(output)
# return output


if __name__ == "__main__":
    generate_prediction()
