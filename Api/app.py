import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os

# ---------------------------------------------------
# Base Directory
# ---------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

# Add root project path
sys.path.append(BASE_DIR)

# ---------------------------------------------------
# Import Neuro Fuzzy Model
# ---------------------------------------------------

from MODEL.fuzzy_model import NeuroFuzzySystem

# ---------------------------------------------------
# Load Model and Scaler
# ---------------------------------------------------

model_path = os.path.join(
    BASE_DIR,
    "MODEL",
    "saved_model.pkl"
)

scaler_path = os.path.join(
    BASE_DIR,
    "MODEL",
    "scaler.pkl"
)

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# ---------------------------------------------------
# Initialize Fuzzy System
# ---------------------------------------------------

fuzzy_system = NeuroFuzzySystem()

# ---------------------------------------------------
# Streamlit Config
# ---------------------------------------------------

st.set_page_config(
    page_title="Neuro-Fuzzy Disease Diagnosis",
    layout="centered"
)

# ---------------------------------------------------
# App Title
# ---------------------------------------------------

st.title("🧠 Neuro-Fuzzy Disease Diagnosis System")

st.write(
    "Enter patient details below to predict diabetes risk."
)

# ---------------------------------------------------
# User Inputs
# ---------------------------------------------------

pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=1
)

glucose = st.number_input(
    "Glucose Level",
    min_value=0,
    max_value=300,
    value=120
)

blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=0,
    max_value=200,
    value=70
)

skin_thickness = st.number_input(
    "Skin Thickness",
    min_value=0,
    max_value=100,
    value=20
)

insulin = st.number_input(
    "Insulin",
    min_value=0,
    max_value=900,
    value=79
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=25.0
)

diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30
)

# ---------------------------------------------------
# Fuzzy Result Function
# ---------------------------------------------------

def get_fuzzy_result(prediction):

    if prediction == 1:
        return "⚠️ High risk detected by Neuro-Fuzzy System"

    return "✅ Low diabetes risk detected"

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

if st.button("Predict"):

    try:

        input_data = np.array([[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]])

        # Scale input
        scaled_data = scaler.transform(input_data)

        # Predict
        prediction = model.predict(scaled_data)

        # Fuzzy output
        fuzzy_result = get_fuzzy_result(prediction[0])

        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.error("Diabetes Positive")

        else:
            st.success("Diabetes Negative")

        st.info(fuzzy_result)

    except Exception as e:
        st.error(f"Prediction Error: {e}")