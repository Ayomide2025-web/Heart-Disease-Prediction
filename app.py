import streamlit as st
import pandas as pd
import joblib

# ==========================================
# LOAD SAVED PIPELINE
# ==========================================

rf_pipeline = joblib.load("rf_pipeline.pkl")

model = rf_pipeline["model"]
scaler = rf_pipeline["scaler"]

# USE EXACT FEATURES FROM TRAINING
features = rf_pipeline["features"]

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# ==========================================
# APP TITLE
# ==========================================

st.title("❤️ Heart Disease Prediction App")

st.write(
    "This application predicts the likelihood of heart disease using a Random Forest Model."
)

st.markdown("---")

# ==========================================
# USER INPUTS
# ==========================================

st.subheader("Enter Patient Information")

age = st.number_input("Age", min_value=1, max_value=120, value=40)

sex = st.selectbox("Sex", ["Male", "Female"])

chest_pain_type = st.selectbox(
    "Chest Pain Type",
    [0, 1, 2, 3]
)

resting_blood_pressure = st.number_input(
    "Resting Blood Pressure",
    min_value=50,
    max_value=300,
    value=120
)

cholesterol = st.number_input(
    "Cholesterol Level",
    min_value=50,
    max_value=700,
    value=200
)

fasting_blood_sugar = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    [0, 1]
)

ecg = st.selectbox(
    "ECG Result",
    [0, 1, 2]
)

max_heart_rate = st.number_input(
    "Maximum Heart Rate",
    min_value=50,
    max_value=250,
    value=150
)

exercise_induced_chest_pain = st.selectbox(
    "Exercise Induced Chest Pain",
    [0, 1]
)

st_depression = st.number_input(
    "ST Depression",
    min_value=0.0,
    max_value=10.0,
    value=1.0
# MATCH EXACT TRAINING FEATURE ORDER
# ==========================================

input_data = input_data.reindex(columns=features)

# ==========================================
# PREDICTION
# ==========================================

if st.button("Predict Heart Disease"):

    try:

        # SCALE INPUT
        scaled_data = scaler.transform(input_data)

        # MAKE PREDICTION
        prediction = model.predict(scaled_data)

        # GET PROBABILITIES
        probability = model.predict_proba(scaled_data)

        st.markdown("---")

        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.error("⚠️ High likelihood of Heart Disease detected.")
        else:
            st.success("✅ Low likelihood of Heart Disease detected.")

        st.subheader("Prediction Probability")

        st.write(
            f"Heart Disease Probability: {probability[0][1] * 100:.2f}%"
        )

        st.write(
            f"No Heart Disease Probability: {probability[0][0] * 100:.2f}%"
        )

    except Exception as e:

        st.error("An error occurred during prediction.")

        st.write("Error Details:")
        st.code(str(e))
