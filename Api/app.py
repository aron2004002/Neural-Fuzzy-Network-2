import gradio as gr
import numpy as np
import joblib
import os
import sys

# ---------------------------------------------------
# Base Directory
# ---------------------------------------------------

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

sys.path.append(BASE_DIR)

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------

model_path = os.path.join(BASE_DIR, "MODEL", "saved_model.pkl")
scaler_path = os.path.join(BASE_DIR, "MODEL", "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# ---------------------------------------------------
# Prediction Function
# ---------------------------------------------------

def predict_diabetes(
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree,
    age
):

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

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)

    if prediction[0] == 1:
        return "⚠️ Diabetes Positive"

    return "✅ Diabetes Negative"

# ---------------------------------------------------
# Gradio Interface
# ---------------------------------------------------

interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Glucose Level"),
        gr.Number(label="Blood Pressure"),
        gr.Number(label="Skin Thickness"),
        gr.Number(label="Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="Diabetes Pedigree Function"),
        gr.Number(label="Age")
    ],
    outputs="text",
    title="🧠 Neuro-Fuzzy Disease Diagnosis System",
    description="Enter patient details to predict diabetes risk."
)

interface.launch()