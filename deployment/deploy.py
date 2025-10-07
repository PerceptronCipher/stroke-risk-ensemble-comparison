import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load model
model = joblib.load(r"C:\Users\USER\My notebook\DataSciencePro\stroke-risk-ensemble-comparison\models\best_model.pkl")

# App title
st.title("🏥 Stroke Risk Prediction")
st.write("Enter patient information to predict stroke risk")

# Input fields
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    ever_married = st.selectbox("Ever Married", ["No", "Yes"])

with col2:
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, max_value=300.0, value=100.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# Predict button
if st.button("Predict Stroke Risk"):
    # Encode inputs (matching training encoding)
    gender_map = {"Male": 1, "Female": 0, "Other": 2}
    married_map = {"No": 0, "Yes": 1}
    work_map = {"Private": 2, "Self-employed": 3, "Govt_job": 0, "children": 4, "Never_worked": 1}
    residence_map = {"Rural": 0, "Urban": 1}
    smoking_map = {"Unknown": 0, "formerly smoked": 1, "never smoked": 2, "smokes": 3}
    hyp_map = {"No": 0, "Yes": 1}
    heart_map = {"No": 0, "Yes": 1}
    
    # Create input array
    input_data = pd.DataFrame({
        'id': [0],
        'gender': [gender_map[gender]],
        'age': [age],
        'hypertension': [hyp_map[hypertension]],
        'heart_disease': [heart_map[heart_disease]],
        'ever_married': [married_map[ever_married]],
        'work_type': [work_map[work_type]],
        'Residence_type': [residence_map[residence_type]],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_map[smoking_status]]
    })
    
    # Scale numerical features (same as training)
    scaler = StandardScaler()
    input_data[['age', 'avg_glucose_level', 'bmi']] = scaler.fit_transform(input_data[['age', 'avg_glucose_level', 'bmi']])
    
    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    # Display results
    st.write("---")
    if prediction == 1:
        st.error(f"HIGH RISK: Stroke probability is {probability[1]*100:.1f}%")
    else:
        st.success(f"LOW RISK: No stroke probability is {probability[0]*100:.1f}%")