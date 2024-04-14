import streamlit as st 
import numpy as np 
import joblib
import os

# Get the absolute path of the current script file
dir = os.path.abspath(__file__)

# Append the parent directory of the current script to the Python path
sys.path.append(os.path.dirname(os.path.dirname(dir)))
path_to_model = '. /RF_loan_model.joblib'



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