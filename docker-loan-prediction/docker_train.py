import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import joblib


# Load dataset
df = pd.read_csv("train.csv")
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

# Log transformation and Domain Processing
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
le = LabelEncoder()
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

model_forest = grid_forest.fit(X_train, y_train)


# predictions = model_forest.predict(X_test)

# accuracy = metrics.accuracy_score(y_test, predictions)
# f1_score = metrics.f1_score(y_test, predictions)
# print(f'Accuracy: {accuracy:.2f}')
# print(f'F1 Score: {f1_score:.2f}')


joblib.dump(model_forest, "RF_loan_model")

print("welcome to the loan prediciton model")
print(
    "Please enter yout details to get to know the status of your application as per below instructions"
)
print("Type 'exit' to terminate..........\n")
print(
    """Gender :Female = 0, male = 1
        Married: No = 0 , Yes 1
        Education : Graduate = 0, under-graduate = 1
        Self_Employed : No = 0, Yes 1
        Property_Area: Urban = 2, semiurban = 1, rural = 0
        Loan_status : No = 0, Yes = 1 \n
"""
)

print(
    """Enter the data with below mentioned order, where values are seprated by comma:
Gender, Married ,Dependents, Education, Self_employed,  Loan_Amount, Loan_Amount_Term, Credit_History,Property_Area,TotalIncome\n"""
)


while True:
    user_data = input("Enter your Details:")

    if user_data == "exit":
        break

    data = list(map(float, user_data.split(",")))  # convert to floats

    # Basic Validation

    if len(data) < 10:
        print("Incomplete data provided !")

    else:

        # Prediciting the value
        predicted_value = model_forest.predict([data])
        print("****************************************************************/")
        if predicted_value == 1:
            print("congratulations! your loan request has been approved")

        else:
            print("sorry! your loan request has been rejected, please try again")
            print("****************************************************************/")