import streamlit as st
import pandas as pd
import joblib

# ==========================================
# LOAD SAVED PIPELINE
# ==========================================

rf_pipeline = joblib.load("rf_pipeline.pkl")

model = rf_pipeline["model"]
scaler = rf_pipeline["scaler"]

# ==========================================
# EXACT FEATURE ORDER USED DURING TRAINING
# ==========================================

features = [
    'age',
    'sex',
    'chest_pain_type',
    'resting_blood_pressure',
    'cholesterol',
    'fasting_blood_sugar',
    'ecg',
    'max_heart_rate',
    'exercise_induced_chest_pain',
    'st_depression',
    'st_slope',
    'stained_blood_vessels',
    'blood_disorder'
]

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# ==========================================
# TITLE
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

age = st.number_input("Age", 1, 120, 40)

sex = st.selectbox("Sex", ["Male", "Female"])

chest_pain_type = st.selectbox("Chest Pain Type", [0, 1, 2, 3])

resting_blood_pressure = st.number_input(
    "Resting Blood Pressure",
    50,
    300,
    120
)

cholesterol = st.number_input(
    "Cholesterol Level",
    50,
    700,
    200
)

fasting_blood_sugar = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    [0, 1]
)

ecg = st.selectbox("ECG Result", [0, 1, 2])

max_heart_rate = st.number_input(
    "Maximum Heart Rate",
    50,
    250,
    150
)

exercise_induced_chest_pain = st.selectbox(
    "Exercise Induced Chest Pain",
    [0, 1]
)

st_depression = st.number_input(
    "ST Depression",
    0.0,
    10.0,
    1.0
)

st_slope = st.selectbox("ST Slope", [0, 1, 2])

stained_blood_vessels = st.selectbox(
    "Number of Stained Blood Vessels",
    [0, 1, 2, 3, 4]
)

blood_disorder = st.selectbox(
    "Blood Disorder",
    [0, 1, 2, 3]
)

# ==========================================
# ENCODE CATEGORICAL VARIABLES
# ==========================================

sex = 1 if sex == "Male" else 0

# ==========================================
# CREATE INPUT DATAFRAME
# ==========================================

input_data = pd.DataFrame([{
    'age': age,
    'sex': sex,
    'chest_pain_type': chest_pain_type,
    'resting_blood_pressure': resting_blood_pressure,
    'cholesterol': cholesterol,
    'fasting_blood_sugar': fasting_blood_sugar,
    'ecg': ecg,
    'max_heart_rate': max_heart_rate,
    'exercise_induced_chest_pain': exercise_induced_chest_pain,
    'st_depression': st_depression,
    'st_slope': st_slope,
    'stained_blood_vessels': stained_blood_vessels,
    'blood_disorder': blood_disorder
}])

# ==========================================
# ENSURE EXACT FEATURE ORDER
# ==========================================

input_data = input_data[features]

# ==========================================
# SCALE INPUT DATA
# ==========================================

scaled_data = scaler.transform(input_data)

# ==========================================
# PREDICTION
# ==========================================

if st.button("Predict Heart Disease"):

    prediction = model.predict(scaled_data)

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
