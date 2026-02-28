import streamlit as st
import joblib
import numpy as np

# ----- Load Model -----
model = joblib.load("loan_model.pkl")

# ----- Optional: Load Scaler if used -----
# scaler = joblib.load("scaler.pkl")  # Uncomment if you used a scaler during training

# ----- UI -----
st.title("🏦 Loan Prediction App")
st.write("Welcome Muskan — your ML app is running ✅")

# --- Step 1: User Inputs (numeric + categorical) ---

# Numeric Inputs
age = st.number_input("Age", 18, 100, 30)
annual_income = st.number_input("Annual Income", 0, 1000000, 50000)
monthly_income = st.number_input("Monthly Income", 0, 100000, 4000)
debt_to_income_ratio = st.number_input("Debt to Income Ratio", 0.0, 100.0, 20.0)
credit_score = st.number_input("Credit Score", 300, 850, 650)
loan_amount = st.number_input("Loan Amount", 0, 1000000, 10000)
interest_rate = st.number_input("Interest Rate (%)", 0.0, 50.0, 10.0)
loan_term = st.number_input("Loan Term (months)", 1, 360, 60)
installment = st.number_input("Installment", 0, 100000, 500)
num_of_open_accounts = st.number_input("Number of Open Accounts", 0, 50, 5)
total_credit_limit = st.number_input("Total Credit Limit", 0, 1000000, 20000)
current_balance = st.number_input("Current Balance", 0, 1000000, 5000)
delinquency_history = st.number_input("Delinquency History", 0, 100, 0)
public_records = st.number_input("Public Records", 0, 50, 0)
num_of_delinquencies = st.number_input("Number of Delinquencies", 0, 50, 0)
income_consistency = st.number_input("Income Consistency", 0, 10, 8)
credit_utilization = st.number_input("Credit Utilization (%)", 0.0, 100.0, 25.0)
installment_income_ratio = st.number_input("Installment to Income Ratio (%)", 0.0, 100.0, 10.0)

# Categorical Inputs
gender = st.selectbox("Gender", ["Male", "Other"])
marital_status = st.selectbox("Marital Status", ["Married", "Single", "Widowed"])
education_level = st.selectbox("Education Level", ["High School", "Master's", "Other", "PhD"])
employment_status = st.selectbox("Employment Status", ["Retired", "Self-employed", "Student", "Unemployed"])
loan_purpose = st.selectbox("Loan Purpose", ["Car", "Debt consolidation", "Education", "Home", "Medical", "Other", "Vacation"])
grade_subgrade = st.selectbox("Grade Subgrade", [
    "A2","A3","A4","A5","B1","B2","B3","B4","B5",
    "C1","C2","C3","C4","C5","D1","D2","D3","D4","D5",
    "E1","E2","E3","E4","E5","F1","F2","F3","F4","F5"
])
age_group = st.selectbox("Age Group", ["Mid", "Senior", "Old"])

# ----- Step 2: Convert Inputs to 70-Feature Array -----
input_data = np.zeros(70)

# Numeric features
input_data[0] = age
input_data[1] = annual_income
input_data[2] = monthly_income
input_data[3] = debt_to_income_ratio
input_data[4] = credit_score
input_data[5] = loan_amount
input_data[6] = interest_rate
input_data[7] = loan_term
input_data[8] = installment
input_data[9] = num_of_open_accounts
input_data[10] = total_credit_limit
input_data[11] = current_balance
input_data[12] = delinquency_history
input_data[13] = public_records
input_data[14] = num_of_delinquencies
input_data[15] = income_consistency
input_data[16] = credit_utilization
input_data[17] = installment_income_ratio

# Categorical features — One-Hot Encoding
# Gender
if gender == "Male":
    input_data[18] = 1
elif gender == "Other":
    input_data[19] = 1

# Marital Status
if marital_status == "Married":
    input_data[20] = 1
elif marital_status == "Single":
    input_data[21] = 1
elif marital_status == "Widowed":
    input_data[22] = 1

# Education Level
if education_level == "High School":
    input_data[23] = 1
elif education_level == "Master's":
    input_data[24] = 1
elif education_level == "Other":
    input_data[25] = 1
elif education_level == "PhD":
    input_data[26] = 1

# Employment Status
if employment_status == "Retired":
    input_data[27] = 1
elif employment_status == "Self-employed":
    input_data[28] = 1
elif employment_status == "Student":
    input_data[29] = 1
elif employment_status == "Unemployed":
    input_data[30] = 1

# Loan Purpose
loan_purpose_dict = {
    "Car": 31, "Debt consolidation": 32, "Education": 33, "Home": 34,
    "Medical": 35, "Other": 36, "Vacation": 37
}
if loan_purpose in loan_purpose_dict:
    input_data[loan_purpose_dict[loan_purpose]] = 1

# Grade Subgrade (indexes 38–66)
grade_dict = {grade: idx for idx, grade in enumerate([
    "A2","A3","A4","A5","B1","B2","B3","B4","B5",
    "C1","C2","C3","C4","C5","D1","D2","D3","D4","D5",
    "E1","E2","E3","E4","E5","F1","F2","F3","F4","F5"
], start=38)}
if grade_subgrade in grade_dict:
    input_data[grade_dict[grade_subgrade]] = 1

# Age Group
if age_group == "Mid":
    input_data[67] = 1
elif age_group == "Senior":
    input_data[68] = 1
elif age_group == "Old":
    input_data[69] = 1

# ----- Step 3 & 4: Prediction Button -----
if st.button("Check"):
    # If scaler used during training:
    # input_data_scaled = scaler.transform(input_data.reshape(1, -1))
    # prediction = model.predict(input_data_scaled)

    prediction = model.predict(input_data.reshape(1, -1))

    if prediction[0] == 1:
        st.success("✅ Loan Likely Approved")
    else:
        st.error("❌ Loan Likely Rejected")
