# Customer Churn Prediction API (Dockerized)

A machine learning-powered API that predicts whether a customer is likely to churn based on historical Telco data. Built with Python, Flask, and Docker — ready for local testing with Postman and deployment.

---

## Project Overview

This API serves a trained Logistic Regression model for predicting customer churn. It accepts JSON input via a `/predict` endpoint and returns:

- `churn_prediction`: 0 or 1  
- `churn_probability`: float (e.g., 0.72)

---

## Dataset

- Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Target: `Churn` (Yes/No → 1/0)
- Features: Demographics, account info, service usage, contract type, etc.

---

## Tech Stack

- **Python 3.11**
- **Flask** (REST API)
- **scikit-learn** (ML model)
- **Pandas** (Data processing)
- **Docker** (Containerization)
- **Postman** (Testing)

---

## How to Run (Locally)
Step 1
''' bash 
https://github.com/Anish-Galgotia/churn-api-project
cd churn-api-project

Step 2:
python train_model.py

Step 3 (Run the API (without Docker)):
pip install -r requirements.txt
python app.py

# How to Run with Docker
1. docker build -t churn-api .
2. docker run -p 8000:8000 churn-api

API: http://localhost:8000/

POST /predict
{
  "tenure": 24,
  "MonthlyCharges": 70.5,
  "TotalCharges": 1500.0,
  "Contract_Two year": 0,
  "Contract_One year": 1,
  "Contract_Month-to-month": 0,
  "InternetService_Fiber optic": 1,
  "PaymentMethod_Electronic check": 1,
  "OnlineSecurity_Yes": 1,
  "OnlineBackup_No": 1,
  "TechSupport_No": 1,
  "StreamingMovies_Yes": 0,
  "PaperlessBilling_Yes": 1,
  "gender_Male": 1,
  "Partner_Yes": 0,
  "Dependents_No": 1
}






