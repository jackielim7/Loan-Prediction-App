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

    # Input User
    person_age = st.slider("Age", 1, 150, 30)
    person_gender = st.selectbox("Gender", ["male", "female"])
    person_education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "Associate", "Doctorate"])
    person_income = st.number_input("Annual Income", 0, 6000000, 50000, step=10000)
    person_emp_exp = st.slider("Years of Employment", 0, 140, 5)
    person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    loan_amnt = st.number_input("Loan Amount", 100, 100000, 5000, step=1000)
    loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION", "PERSONAL"])
    loan_int_rate = st.number_input("Interest Rate (%)", 0.0, 50.0, 10.0, step=0.1)
    loan_percent_income = st.number_input("Loan Percent Income", 0.0, 1.0, 0.2, step=0.01)
    cb_person_cred_hist_length = st.slider("Credit History Length (years)", 0, 30, 5)
    credit_score = st.slider("Credit Score", 300, 900, 600)
    previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults", ["Yes", "No"])

    # Simpan input asli user (belum di-encode)
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

    # Tampilkan data input dari user
    st.write("### Your Input")
    st.dataframe(pd.DataFrame([raw_data]))

    if st.button("Predict Loan Approval"):
        try:
            model = load_model("best_model.pkl")
            transformer = load_model("transformer.pkl")
            label_encoders = load_model("label_encoders.pkl")
            pred_class, pred_proba, class_labels = predict_with_model(model, transformer, label_encoders, raw_data)

            # Tampilkan prediksi
            st.success(f"Prediction: `{pred_class}`")

            # Tampilkan probabilitas untuk masing-masing kelas
            proba_df = pd.DataFrame([pred_proba], columns=class_labels).round(4)
            st.write("### Class Probabilities")
            st.dataframe(proba_df)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()
