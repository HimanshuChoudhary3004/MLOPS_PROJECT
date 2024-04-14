{
    "dataframe_split": {
        "columns": [
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History",
            "Property_Area",
            "TotalIncome"
        ],
        "data": [
            [
                1,
                0,
                0,
                0,
                0,
                4.9875,
                360.0,
                1.0,
                2.0,
                8.698
            ]
        ]
    }
}



# Tracking cmd
mlflow run . --experiment-name loan_prediction --env-manager local

# Rest API on port 9000
mlflow models serve -m mlflow models serve -m mlartifacts/509308075366814341/557b733023484d5c8f66e7c2ded9cf17/artifacts/LogisticRegression --port 9000

# Getting predictions from rest api at port 9000
http://127.0.0.1:9000//invocations

# Storing Experiment data in database
mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri mysql://root:password@localhost/mlflow_db --default-artifact-root $PWD/mlruns

# serving the model from model registry

set MLFLOW_TRACKING_URI=http://127.0.0.1:5001


mlflow models serve -m "models:/Loan_pred_Log_reg/Production" -p 1234
http://127.0.0.1:1234//invocations



