# 🏦 Loan Approval Prediction System (Flask + Machine Learning)

An end-to-end Machine Learning project that predicts whether a loan application will be approved or not based on applicant details. The system is deployed using a Flask web application for real-time predictions.

---

## 📌 Project Overview

Loan approval is a crucial process in banking and financial institutions. This project automates the loan approval process using Machine Learning algorithms by analyzing applicant financial and demographic data.

The system helps in:
- Reducing manual effort
- Improving decision accuracy
- Speeding up loan approval process

---

## 🚀 Features

✅ Data Cleaning & Preprocessing  
✅ Exploratory Data Analysis (EDA)  
✅ Feature Engineering  
✅ Model Training (Logistic Regression, Random Forest)  
✅ Model Evaluation (Accuracy, Precision, Recall, F1-score)  
✅ Confusion Matrix Analysis  
✅ Flask Web Application for Live Prediction  

---

## 🛠 Tech Stack

- Python  
- Pandas & NumPy  
- Matplotlib & Seaborn  
- Scikit-learn  
- Flask  
- HTML, CSS  
- Joblib  
- Git & GitHub  

---

## 📊 Dataset Overview

- Total Records: ~20,000  
- Target Variable: `loan_paid_back` (1 = Approved, 0 = Rejected)  

### 🔹 Features Used:

- **Demographic Data:**  
  - Age  
  - Gender  
  - Marital Status  
  - Education Level  

- **Financial Data:**  
  - Annual Income  
  - Monthly Income  
  - Credit Score  

- **Loan Details:**  
  - Loan Amount  
  - Loan Term  
  - Installment  

- **Credit Behavior:**  
  - Debt-to-Income Ratio  
  - Delinquency History  
  - Public Records  

---

## 📈 Machine Learning Workflow

1. Data Collection  
2. Data Cleaning  
3. Exploratory Data Analysis (EDA)  
4. Feature Engineering  
5. Encoding Categorical Variables  
6. Train-Test Split  
7. Model Training  
   - Logistic Regression  
   - Random Forest  
8. Model Evaluation  
9. Model Selection (Random Forest - ~89% Accuracy)  
10. Model Saving using Joblib  
11. Deployment using Flask  

---

## 📁 Project Structure
loan_approval/
│
├── app/
│ ├── app.py
│ ├── loan_model.pkl
│ ├── templates/
│ │ └── index.html
│
├── data/
├── notebook/
└── README.md


---

## ▶️ How to Run the Project

### Step 1: Install Dependencies
pip install -r requirements.txt

### Step 2: Run Flask App
cd app
python app.py

### Step 3: Open in Browser
http://127.0.0.1:5000/



---

## 📊 Model Performance

- Logistic Regression Accuracy: ~88%  
- Random Forest Accuracy: ~89%  

### Classification Insights:
- High precision for loan approval prediction  
- Strong recall for identifying approved loans  
- Balanced performance across classes  

---

## 🔮 Future Scope

- Deploy on cloud platforms (Render / AWS)  
- Add user authentication system  
- Improve UI/UX design  
- Integrate real-time database  
- Use advanced models like XGBoost  

---

## 👩‍💻 Author

**Muskan**  
MCA Student | Aspiring Data Scientist 🚀  

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
