
import mlflow
model_name = 'Loan_pred_Log_reg'

stage = "Production"

mlflow.set_tracking_uri("http://127.0.0.1:5001/")

# Load model as a PyFuncModel
loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")

data = [[1, 0, 0, 0, 0, 4.9875, 360.0, 1.0, 2.0, 8.698]]

# Predict on a Pandas DataFrame.
import pandas as pd

print(loaded_model.predict(pd.DataFrame(data)))