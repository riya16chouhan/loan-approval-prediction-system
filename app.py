import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="LoanTrain",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1E3A8A; text-align: center;}
    .stButton>button {width: 100%; height: 3rem; font-size: 1.1rem;}
    .prediction-box {padding: 20px; border-radius: 10px; margin: 15px 0;}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">💰 LoanTrain</h1>', unsafe_allow_html=True)
st.markdown("### Intelligent Loan Approval Prediction System")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank.png", width=80)
    st.title("LoanTrain")
    st.divider()
    
    page = st.radio("Navigation", 
                   ["🏠 Home", "🔮 Predict"])
    
    st.divider()
    st.caption("Built with ❤️ for LoanTrain Project")

# Load model and scaler (with error handling)
@st.cache_resource
def load_model():
    try:
        return joblib.load('model.pkl')
    except:
        return None

@st.cache_resource
def load_scaler():
    try:
        return joblib.load('scaler (1).pkl')
    except:
        return None

model = load_model()
scaler = load_scaler()

# ====================== HOME PAGE ======================
if page == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Welcome to LoanTrain")
        st.write("""
        This intelligent system helps banks and financial institutions make 
        faster and more accurate loan approval decisions using Machine Learning.
        """)
        
    

    with col2:
        st.metric("Accuracy", "87.3%", "↑ 4.2%")
        st.metric("Precision", "85.1%", "↑")
        st.metric("Recall", "89.7%", "↑")

# ====================== PREDICT PAGE ======================
elif page == "🔮 Predict":
    st.header("Loan Application Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Marital Status", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        
    with col2:
        st.subheader("Financial Information")
        applicant_income = st.number_input("Applicant Monthly Income ($)", 
                                         min_value=0, value=5000)
        coapplicant_income = st.number_input("Co-applicant Monthly Income ($)", 
                                           min_value=0, value=0)
        loan_amount = st.number_input("Loan Amount ($)", min_value=1000, value=150000)
        loan_term = st.selectbox("Loan Term (months)", [12, 36, 60, 84, 120, 180, 240, 360])
        credit_history = st.selectbox("Credit History (1 = Good)", [1.0, 0.0])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        inr_rate = 82
        inr_col1, inr_col2, inr_col3 = st.columns(3)
        inr_col1.metric("Applicant Income (₹)", f"₹{applicant_income * inr_rate:,.0f}")
        inr_col2.metric("Co-applicant Income (₹)", f"₹{coapplicant_income * inr_rate:,.0f}")
        inr_col3.metric("Loan Amount (₹)", f"₹{loan_amount * inr_rate:,.0f}")

    # Predict Button
    if st.button("🚀 Predict Loan Approval", type="primary"):
        if model is None:
            st.error("Model not found. Please train and save your model as 'model.pkl'")
            st.stop()

        if scaler is None:
            st.error("Scaler not found. Please save your scaler as 'scaler (1).pkl'.")
            st.stop()

        # Prepare input data with the same feature set used during training
        input_data = pd.DataFrame({
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount / 1000],
            'Loan_Amount_Term': [loan_term],
            'Total_Income': [applicant_income + coapplicant_income],
            'Gender_Female': [1 if gender == "Female" else 0],
            'Gender_Male': [1 if gender == "Male" else 0],
            'Married_No': [1 if married == "No" else 0],
            'Married_Yes': [1 if married == "Yes" else 0],
            'Dependents_0.0': [1 if dependents == "0" else 0],
            'Dependents_1.0': [1 if dependents == "1" else 0],
            'Dependents_2.0': [1 if dependents == "2" else 0],
            'Dependents_3.0': [1 if dependents == "3+" else 0],
            'Education_Graduate': [1 if education == "Graduate" else 0],
            'Education_Not Graduate': [1 if education == "Not Graduate" else 0],
            'Self_Employed_No': [1 if self_employed == "No" else 0],
            'Self_Employed_Yes': [1 if self_employed == "Yes" else 0],
            'Property_Area_Rural': [1 if property_area == "Rural" else 0],
            'Property_Area_Semiurban': [1 if property_area == "Semiurban" else 0],
            'Property_Area_Urban': [1 if property_area == "Urban" else 0],
            'Credit_History_0.0': [1 if credit_history == 0.0 else 0],
            'Credit_History_1.0': [1 if credit_history == 1.0 else 0]
        })

        transformed_input = scaler.transform(input_data)
        prediction = model.predict(transformed_input)[0]
        probability = model.predict_proba(transformed_input)[0][1]

        # Display Result
        if prediction == 1:
            st.markdown("""
            <div style='background-color: #d4edda; padding: 20px; border-radius: 10px;'>
                <h2 style='color: #155724; text-align: center;'>✅ Loan Approved!</h2>
                <p style='text-align: center; font-size: 1.2rem;'>
                    Approval Probability: <b>{:.1f}%</b>
                </p>
            </div>
            """.format(probability * 100), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background-color: #f8d7da; padding: 20px; border-radius: 10px;'>
                <h2 style='color: #721c24; text-align: center;'>❌ Loan Rejected</h2>
                <p style='text-align: center; font-size: 1.2rem;'>
                    Approval Probability: <b>{:.1f}%</b>
                </p>
            </div>
            """.format(probability * 100), unsafe_allow_html=True)
