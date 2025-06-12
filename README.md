# churnPredictor

A machine learning model to predict customer churn for a subscription-based service using historical customer data.

## 🔍 Overview
This project uses customer demographics and usage behavior to predict whether a customer will leave the service. Models used:
- Logistic Regression
- Random Forest
- Gradient Boosting

## 📁 Dataset
Dataset: `Churn_Modelling.csv`  
Features include credit score, age, tenure, balance, number of products, and geography.

## 🛠️ Tech Stack
- Python
- pandas, scikit-learn

## 🚀 How to Run
1. Clone the repo
2. Place the dataset in the root directory
3. Run the script:
   ```bash
   pip install pandas scikit-learn
   python churn_prediction.py
