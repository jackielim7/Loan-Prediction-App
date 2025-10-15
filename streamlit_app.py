import streamlit as st
import pandas as pd
import joblib

def load_model(filename):
    return joblib.load(filename)

def predict_with_model(model, transformer, label_encoders, input_data):
    df = pd.DataFrame([input_data])
    for col, encoder in label_encoders.items():
        df[col] = encoder.transform(df[col].astype(str))
    X = transformer.transform(df)
    pred = model.predict(X)
    proba = model.predict_proba(X)
    return pred[0], proba[0], model.classes_

def main():
    st.title("Loan Approval Prediction")
    st.info("This app predicts whether your loan will be approved.")

    #Input User
    person_age = st.slider("Age", min_value = 1, max_value = 150, value = 1)
    person_gender = st.selectbox("Gender", ["male", "female"])
    person_education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "Associate", "Doctorate"])
    person_income = st.number_input("Annual Income", min_value = 0, max_value = 6000000, value = 50000, step = 10000)
    person_emp_exp = st.slider("Years of Employment", min_value = 0, max_value = 140, value = 1)
    person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    loan_amnt = st.number_input("Loan Amount", min_value = 100, max_value = 100000, value = 5000, step = 1000)
    loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION", "PERSONAL"])
    loan_int_rate = st.number_input("Interest Rate (%)", min_value = 0.0, max_value = 50.0, value = 10.0, step=0.1)
    loan_percent_income = st.number_input("Loan Percent Income",  min_value = 0.0, max_value = 1.0, value =  0.2, step=0.01)
    cb_person_cred_hist_length = st.slider("Credit History Length (years)", min_value = 0,  max_value = 30, value = 1)
    credit_score = st.slider("Credit Score", min_value = 300, max_value = 900, value = 600)
    previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults", ["Yes", "No"])


    #Save input from user (Not Encoded)
    raw_data = {
        'person_age': person_age,
        'person_income': person_income,
        'person_emp_exp': person_emp_exp,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'credit_score': credit_score,
        'person_gender': person_gender,
        'previous_loan_defaults_on_file': previous_loan_defaults_on_file,
        'person_home_ownership': person_home_ownership,
        'person_education': person_education,
        'loan_intent': loan_intent
    }

    #Show Inputed Data
    st.write("### Your Input")
    st.dataframe(pd.DataFrame([raw_data]))

    if st.button("Predict Loan Approval"):
        try:
            model = load_model("best_model.pkl")
            transformer = load_model("transformer.pkl")
            label_encoders = load_model("label_encoders.pkl")
            pred_class, pred_proba, class_labels = predict_with_model(model, transformer, label_encoders, raw_data)

            #Show Prediction
            st.success(f"Prediction: `{pred_class}`")

            #Show All Clasess Probability
            proba_df = pd.DataFrame([pred_proba], columns=class_labels).round(4)
            st.write("### Class Probabilities")
            st.dataframe(proba_df)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()
