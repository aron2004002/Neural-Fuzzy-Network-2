import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os


# Fix import path
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

from MODEL.fuzzy_model import NeuroFuzzySystem


# Load trained model
model = joblib.load("MODEL/saved_model.pkl")
scaler = joblib.load("MODEL/scaler.pkl")


# Initialize fuzzy system
fuzzy_system = NeuroFuzzySystem()


# Streamlit page config
st.set_page_config(
    page_title="Neuro-Fuzzy Disease Diagnosis",
    layout="centered"
)


st.title("🧠 Neuro-Fuzzy Disease Diagnosis System")

st.write("Enter patient details below")


# -----------------------------
# User Inputs
# -----------------------------

pregnancies = st.number_input("Pregnancies", 0, 20, 1)

glucose = st.number_input("Glucose Level", 0, 300, 120)

blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)

skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)

insulin = st.number_input("Insulin", 0, 900, 79)

bmi = st.number_input("BMI", 0.0, 70.0, 25.0)

diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    0.0,
    3.0,
    0.5
)

age = st.number_input("Age", 1, 120, 30)


# -----------------------------
# Fuzzy Result Function
# -----------------------------

def get_fuzzy_result(prediction):

    if prediction == 1:
        return "⚠️ High risk detected by Neuro-Fuzzy System"
    else:
        return "✅ Low diabetes risk detected"


# -----------------------------
# Predict Button
# -----------------------------

if st.button("Predict"):

    input_data = [[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]]

    # Scale input
    scaled_data = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(scaled_data)

    # Fuzzy Output
    fuzzy_result = get_fuzzy_result(prediction[0])

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("Diabetes Positive")
    else:
        st.success("Diabetes Negative")

    st.info(fuzzy_result)