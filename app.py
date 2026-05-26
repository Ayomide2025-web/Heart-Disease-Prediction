import streamlit as st
import pandas as pd
import joblib

# ==========================================
# LOAD SAVED PIPELINE
# ==========================================

rf_pipeline = joblib.load("rf_pipeline.pkl")

model = rf_pipeline["model"]
scaler = rf_pipeline["scaler"]
features = rf_pipeline["features"]
# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Heart Disease Prediction App",
    page_icon="❤️",
    layout="centered"
)

# ==========================================
# APP TITLE
# ==========================================

st.title("❤️ Heart Disease Prediction App")

st.write(
    """
    This application predicts the likelihood of heart disease
    using a Random Forest Machine Learning model.
    """
)

st.markdown("---")

# ==========================================
# USER INPUTS
# ==========================================

st.subheader("Enter Patient Information")

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=40
)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

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
)

st_slope = st.selectbox(
    "ST Slope",
    [0, 1, 2]
)

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

input_values = [
    age,
    sex,
    chest_pain_type,
    resting_blood_pressure,
    cholesterol,
    fasting_blood_sugar,
    ecg,
    max_heart_rate,
    exercise_induced_chest_pain,
    st_depression,
    st_slope,
    stained_blood_vessels,
    blood_disorder
]

input_data = pd.DataFrame([input_values])

input_data.columns = features


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

        st.error(
            "⚠️ High likelihood of Heart Disease detected."
        )

    else:

        st.success(
            "✅ Low likelihood of Heart Disease detected."
        )

    st.subheader("Prediction Probability")

    st.write(
        f"Heart Disease Probability: {probability[0][1] * 100:.2f}%"
    )

    st.write(
        f"No Heart Disease Probability: {probability[0][0] * 100:.2f}%"
    )
