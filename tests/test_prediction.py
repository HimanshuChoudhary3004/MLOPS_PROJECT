import pytest
from prediction_model.config import config
from prediction_model.processing.data_handling import load_pipeline, load_dataset
from prediction_model.predict import generate_prediction

# it should not be a null value
# should be a string
# output is 'Y'
# fixtures ----function before test function ---> ensure single_prediction


@pytest.fixture  # Decorator to make sure single_prediction get executed before test function every time
def single_prediction():
    test_data = load_dataset(config.TEST_FILE)
    single_row = test_data[:1]
    result = generate_prediction(single_row)
    return result


# Test functions


def test_single_pred_none(single_prediction):
    assert single_prediction is not None


def test_single_pred_str(single_prediction):
    assert isinstance(single_prediction.get("Prediction")[0], str)


def test_single_pred_output(single_prediction):
    assert single_prediction.get("Prediction")[0] == "Y"
