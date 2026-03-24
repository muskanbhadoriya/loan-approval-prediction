from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("loan_model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # -------- Numeric Inputs --------
        age = float(request.form['age'])
        annual_income = float(request.form['annual_income'])
        monthly_income = float(request.form['monthly_income'])
        debt_to_income_ratio = float(request.form['debt_to_income_ratio'])
        credit_score = float(request.form['credit_score'])
        loan_amount = float(request.form['loan_amount'])
        interest_rate = float(request.form['interest_rate'])
        loan_term = float(request.form['loan_term'])
        installment = float(request.form['installment'])
        num_of_open_accounts = float(request.form['num_of_open_accounts'])
        total_credit_limit = float(request.form['total_credit_limit'])
        current_balance = float(request.form['current_balance'])
        delinquency_history = float(request.form['delinquency_history'])
        public_records = float(request.form['public_records'])
        num_of_delinquencies = float(request.form['num_of_delinquencies'])
        income_consistency = float(request.form['income_consistency'])
        credit_utilization = float(request.form['credit_utilization'])
        installment_income_ratio = float(request.form['installment_income_ratio'])

        # -------- Categorical Inputs --------
        gender = request.form['gender']
        marital_status = request.form['marital_status']
        education_level = request.form['education_level']
        employment_status = request.form['employment_status']
        loan_purpose = request.form['loan_purpose']
        grade_subgrade = request.form['grade_subgrade']
        age_group = request.form['age_group']

        # -------- Create Input Array --------
        input_data = np.zeros(70)

        # Numeric
        input_data[0:18] = [
            age, annual_income, monthly_income, debt_to_income_ratio,
            credit_score, loan_amount, interest_rate, loan_term,
            installment, num_of_open_accounts, total_credit_limit,
            current_balance, delinquency_history, public_records,
            num_of_delinquencies, income_consistency,
            credit_utilization, installment_income_ratio
        ]

        # -------- One-Hot Encoding --------

        # Gender
        if gender == "Male":
            input_data[18] = 1
        elif gender == "Female":
            input_data[19] = 1

        # Marital
        mapping = {"Married":20, "Single":21, "Widowed":22}
        input_data[mapping[marital_status]] = 1

        # Education
        mapping = {"High School":23, "Master's":24, "Other":25, "PhD":26}
        input_data[mapping[education_level]] = 1

        # Employment
        mapping = {"Retired":27, "Self-employed":28, "Student":29, "Unemployed":30}
        input_data[mapping[employment_status]] = 1

        # Loan Purpose
        loan_dict = {
            "Car":31,"Debt consolidation":32,"Education":33,"Home":34,
            "Medical":35,"Other":36,"Vacation":37
        }
        input_data[loan_dict[loan_purpose]] = 1

        # Grade
        grades = [
            "A2","A3","A4","A5","B1","B2","B3","B4","B5",
            "C1","C2","C3","C4","C5","D1","D2","D3","D4","D5",
            "E1","E2","E3","E4","E5","F1","F2","F3","F4","F5"
        ]
        grade_dict = {g:i+38 for i,g in enumerate(grades)}
        input_data[grade_dict[grade_subgrade]] = 1

        # Age Group
        mapping = {"Mid":67, "Senior":68, "Old":69}
        input_data[mapping[age_group]] = 1

        # -------- Prediction --------
        prediction = model.predict(input_data.reshape(1, -1))

        result = "Loan Approved ✅" if prediction[0] == 1 else "Loan Rejected ❌"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)