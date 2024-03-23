import os
import pathlib
import prediction_model

PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent

DATAPATH = os.path.join(PACKAGE_ROOT, "dataset")

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT, "trained_models")
MODEL_NAME = "classification.pkl"

TARGET = "Loan_Status"

# Final set of features
FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
    "CoapplicantIncome",
]


NUM_FEATURES = ["ApplicantIncome", "LoanAmount", "Loan_Amount_Term"]

CAT_FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Credit_History",
    "Property_Area",
]

# in our case feature to encode are same as categorical features
FEATURES_TO_ENCODE = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Credit_History",
    "Property_Area",
]

FEATURE_TO_MODIFY = ["ApplicantIncome"]
FEATURE_TO_ADD = "CoapplicantIncome"
DROP_FEATURE = ["CoapplicantIncome"]

LOG_FEATURES = ["ApplicantIncome", "LoanAmount"]  # Features to take log
