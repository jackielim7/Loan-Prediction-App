# 📦 Loan Prediction App

A Streamlit web application that predicts whether a loan application will be approved based on applicant information such as income, credit score, employment experience, and other relevant factors.
This project was developed as part of my Midterm Exam for the Model Deployment course. The backend model was built using XGBoost, with proper data preprocessing, feature encoding, and model tuning

## Demo App
You can try the live demo here:
[Loan Prediction App](https://md-uts.streamlit.app/)

## Project Overview

The goal of this project is to predict loan approval status using a dataset of applicant and loan information.
The workflow includes:
1. Data Cleaning – handling missing values, fixing inconsistent data, and checking outliers.
2. Feature Engineering & Encoding – transforming categorical and numerical variables using:
   - LabelEncoder for binary columns
   - OneHotEncoder for nominal features
   - OrdinalEncoder for ordered features
   - RobustScaler for numerical features
3. Modeling – training an XGBoost Classifier to predict loan approval status.
4. Hyperparameter Tuning – optimizing model parameters using GridSearchCV.
5. Model Deployment – deploying the trained model using Streamlit with serialized .pkl files.

## Saved Files
| File                 | Description                                |
| -------------------- | ------------------------------------------ |
| `best_model.pkl`     | Trained XGBoost model                      |
| `transformer.pkl`    | Preprocessing pipeline (scalers, encoders) |
| `label_encoders.pkl` | Encoders for categorical features          |


## Author
Jackie Lim
📧 [linkedin.com/in/jackie-lim7/]  
🎓 Machine Learning Midterm Project — BINUS University  
