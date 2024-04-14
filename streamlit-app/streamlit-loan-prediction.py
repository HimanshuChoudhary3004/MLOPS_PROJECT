import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import joblib


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

joblib.dump(model_forest, "RF_loan_model.joblib")

loaded_model = joblib.load('RF_loan_model.joblib')

data = [[
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    4.9876,
    360.0,
    1.0,
    2.0,
    8.698,
]]

print(f"Prediction is : {loaded_model.predict(pd.DataFrame(data))}")

















import streamlit as st 
import numpy as np 
import joblib
import os

# Constructing the full file path to the model file
model_file_path = os.path.join(os.getcwd(), 'RF_loan_model.joblib')

try:
    # Attempting to load the model using the full file path
    model = joblib.load(model_file_path)
    st.write("Model loaded successfully!")
except Exception as e:
    # Print out any error messages or exceptions that occur during the loading process
    st.error(f"Error loading the model: {e}")



def prediction(Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,
        Credit_History,Property_Area):
        # predicting the loan status based on the input data

        if Gender == 'Male':
            Gender = 1
        else :
            Gender = 0
        
        if Married == 'Yes':
            Married = 1
        else :
            Married = 0
        
        if Education == 'Graduate':
            Education = 0
        else :
            Education = 1
        
        if Self_Employed == 'Yes':
            Self_Employed = 1
        else :
            Self_Employed = 0
        
        if Credit_History == 'Outstanding Loan':
            Credit_History = 1
        else :
            Credit_History = 0
        
        if Property_Area == 'Rural':
            Property_Area = 0
        elif Property_Area == 'Semi Urban':
            Property_Area = 1
        else :
            Property_Area = 2
        
        TotalIncome = np.log(ApplicantIncome + CoapplicantIncome)

        predictions = model.predict([[Gender,Married,Dependents,Education,Self_Employed,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area,TotalIncome]])
              
        if predictions == 0:
            return 'Rejected'
        else:
            return 'Approved'


        

def main():
    # Front end
    st.title("Welcome to loan prediction application.")
    st.write("This application will help you to predict whether a loan will be paid off or not.")
    st.header("Please enter your details to proceed with your loan application status.")
    
    Gender = st.selectbox("Gender",('Male','Female'))
    Married = st.selectbox("Married",('Yes','No'))
    Dependents = st.number_input("Dependents")
    Education = st.selectbox("Education",('Graduate','Not Graduate'))
    Self_Employed = st.selectbox("Self_Employed",('Yes','No'))
    ApplicantIncome = st.number_input("Applicant Income")
    CoapplicantIncome = st.number_input("Coapplicant Income")
    LoanAmount = st.number_input("Loan Amount")
    Loan_Amount_Term = st.number_input("Loan_Amount_Term")
    Credit_History = st.selectbox("Credit_History",('Outstanding Loan','No Outstanding Loan'))
    Property_Area = st.selectbox("Property_Area",('Rural','semi-urban','urban'))

    if st.button("Predict"):
        result = prediction(Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,
        Credit_History,Property_Area)

        if result == "Approved":
            st.success("your Loan application is approved")
        else:
            st.error("your Loan application  is Rejected")

if __name__ == "__main__":
    main()