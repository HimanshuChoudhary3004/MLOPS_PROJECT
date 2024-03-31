import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import mlflow
import os


def loan_prediction():
    # Load dataset
    df = pd.read_csv(r"E:\DATA\nachiket_mlops_loan dataset\train.csv")
    numerical_columns = df.select_dtypes(include=["float", "integer"]).columns.to_list()
    categorical_columns = df.select_dtypes(include=["object"]).columns.to_list()
    categorical_columns.remove("Loan_Status")
    categorical_columns.remove("Loan_ID")

    # Filling the category column with mode
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode().iloc[0])

    # Filling in Numerical columns with median values
    for col in numerical_columns:
        df[col] = df[col].fillna(df[col].median())

    # Taking care of outliers
    df[numerical_columns] = df[numerical_columns].apply(
        lambda x: x.clip(*x.quantile([0.5, 0.95]))
    )

    # Log transformationa and Domain Processing
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["TotalIncome"] = pd.to_numeric(df["TotalIncome"], errors="coerce").fillna(0)
    df["TotalIncome"] = np.log(df["TotalIncome"])
    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
    df["LoanAmount"] = pd.to_numeric(df["LoanAmount"], errors="coerce").fillna(0)
    df["LoanAmount"] = np.log(df["LoanAmount"])

    # Dropping ApplicationsIncome and CoapplicantIncome columns
    df = df.drop(columns=["ApplicantIncome", "CoapplicantIncome"])

    # Label Encoding categorical columns
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Encoding target columns
    df["Loan_Status"] = le.fit_transform(df["Loan_Status"])

    # Train test split
    X = df.drop(["Loan_Status", "Loan_ID"], axis=1)
    y = df["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Random Forest classification

    RF = RandomForestClassifier()
    param_grid_rf = {
        "n_estimators": [200, 400, 700],
        "criterion": ["gini", "entropy"],
        "max_depth": [10, 20, 30],
        "max_leaf_nodes": [50, 100],
    }

    grid_forest = GridSearchCV(
        estimator=RF,
        param_grid=param_grid_rf,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        return_train_score=True,
        verbose=0,
    )

    grid_forest.fit(X_train, y_train)

    # Logistic Regression classifier
    LR = LogisticRegression(random_state=42)

    param_grid_lr = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],
    }

    Log_model = GridSearchCV(
        estimator=LR,
        param_grid=param_grid_lr,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        return_train_score=True,
        verbose=0,
    )

    Log_model.fit(X_train, y_train)

    # Decison Tree classifier

    DT = DecisionTreeClassifier(random_state=42)
    param_grid_dt = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 7, 9, 11, 13],
    }

    DT_model = GridSearchCV(
        estimator=DT,
        param_grid=param_grid_dt,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        return_train_score=True,
        verbose=0,
    )

    DT_model.fit(X_train, y_train)

    # Setting Experiment

    mlflow.set_experiment("loan_prediction")

    # Model Evalution Metrics
    def eval(actual, predicted):
        accuracy = metrics.accuracy_score(actual, predicted)
        f1 = metrics.f1_score(actual, predicted)
        fpr, tpr, _ = metrics.roc_curve(actual, predicted)
        auc = metrics.auc(fpr, tpr)
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color="blue", label="ROC curve area =%.2f" % auc)
        plt.plot([0, 1], [0, 1], linestyle="--", color="black")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="upper left")
        # save plot
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/ROC_curve.png")
        # close plot
        plt.close()
        return (accuracy, f1, auc)

    def mlflow_logging(model, X, y, name):

        with mlflow.start_run() as run:
            run = mlflow.active_run()
            run_id = run.info.run_id
            mlflow.set_tag("run_id", run_id)
            pred = model.predict(X)
            # metric
            (accuracy, f1, auc) = eval(y, pred)
            # Logging best parameters from GridSearch
            # mlflow.log_params("f'{model}Best_parametet",model.best_params_)
            # Log the metrics
            mlflow.log_metric("Mean CV Score", model.best_score_)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("auc", auc)

            # Logging artifacts and models
            mlflow.log_artifacts("plots")

            mlflow.sklearn.log_model(model, name)
            print(f"{name} accuracy: {accuracy} f1: {f1} auc: {auc}")

            mlflow.end_run()

    mlflow_logging(grid_forest, X_test, y_test, "RandomForestClassifier")
    mlflow_logging(Log_model, X_test, y_test, "LogisticRegression")
    mlflow_logging(DT_model, X_test, y_test, "DecisionTree")


if __name__ == "__main__":
    loan_prediction()
