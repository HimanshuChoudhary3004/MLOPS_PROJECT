# Importing dependencies
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib
import numpy as np
import pandas as pd


app = FastAPI()
model_name = 'RF_loan_model.joblib'
model = joblib.load(model_name)


#Perform Parsing

class Loan(BaseModel):
    Gender: float
    Married: float
    Dependents: float
    Education: float
    Self_Employed: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: float
    TotalIncome: float


@app.get("/")
def index():
    return {"Welcome to the loan prediction app"}

# Defining the function which will make predictions when the user sends data

@app.post("/predict")
def predict_loan_status(loan_details: Loan):
    data = loan_details.dict()
    gender = data["Gender"]
    married = data["Married"]
    dependents = data["Dependents"]
    education = data["Education"]
    self_employed = data["Self_Employed"]
    loan_amount = data["LoanAmount"]
    loan_amount_term = data["Loan_Amount_Term"]
    credit_history = data["Credit_History"]
    property_area = data["Property_Area"]
    income = data["TotalIncome"]

    # Making predictions
    predictions = model.predict([[gender,married,dependents,education,self_employed,loan_amount,loan_amount_term,credit_history,property_area,income]])

    if predictions == 0:
        pred = 'Rejected'
    else:
        pred = "Approved"

    return {'status of loan application': pred}


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)









