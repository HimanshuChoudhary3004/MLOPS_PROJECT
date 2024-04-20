from flask import Flask, render_template,request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Loading model 
model_name = 'RF_loan_model.joblib'
model =  joblib.load(model_name)

@app.route("/")
def home():
    return render_template('homepage.html')


@app.route("/predict",methods=['POST'])
def predict():

    if request.method == 'POST':
        request_data = dict(request.form)
        del request_data['First_Name']
        del request_data['Last_Name']
        request_data = {k:int(v) for k,v in request_data.items()}
        data = pd.DataFrame([request_data])
        data['TotalIncome'] = data['Applicant_income'] + data['Co_applicant_income']
        data['TotalIncome'] = np.log(data['TotalIncome']).copy()
        data = data[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area','TotalIncome']]
        predictions = model.predict(data)
        prediction_value = predictions[0]

        if int(prediction_value) == 0:
            result = 'Sorry! your loan approval request is Rejected'
        if int(prediction_value) == 1:
            result = 'Congratulations! your loan Approval request is Successful'

        return render_template('homepage.html',prediction = result)

@app.errorhandler(500)
def internal_error(error):
    return "500 : something went wrong"


if __name__=='__main__':
    app.run(host= 'localhost', port=80)
