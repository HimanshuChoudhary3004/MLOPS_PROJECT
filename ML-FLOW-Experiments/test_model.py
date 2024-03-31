data = [[1, 0, 0, 0, 0, 4.9875, 360.0, 1.0, 2.0, 8.698]]


import mlflow
logged_model = 'runs:/0de6a48dd0cb4b54bf5ee1406da90181/LogisticRegression'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
print(loaded_model.predict(pd.DataFrame(data)))

