import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.preprocessing import StandardScaler

# ------------------------------
# Load model and scaler
# ------------------------------
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

# ------------------------------
# Load data (for EDA)
# ------------------------------
df = pd.read_csv("cardio_cleaned.csv")  # Ensure this is cleaned

# ------------------------------
# App Layout
# ------------------------------
st.set_page_config(page_title="Cardiovascular Disease Predictor", layout="wide")
st.title("🫀 Cardiovascular Disease Prediction Dashboard")

st.markdown("""
Welcome to the **Cardiovascular Disease Prediction App**!  
This tool predicts the risk of cardiovascular disease based on patient health inputs.  
The model is trained using Logistic Regression on real-world health data.
""")

st.subheader("Model Details")
st.markdown("""
The model used is a **Logistic Regression** classifier trained on features like age, BP, cholesterol, glucose, BMI, and lifestyle habits.
""")

# ------------------------------
# Sidebar: User Inputs
# ------------------------------
st.sidebar.header("Enter Patient Data")

age_years = st.sidebar.slider("Age (years)", 18, 80, 50)
height_feet = st.sidebar.slider("Height (feet)", 4.5, 7.0, 5.5)
weight = st.sidebar.slider("Weight (kg)", 40, 150, 70)
ap_hi = st.sidebar.slider("Systolic BP (ap_hi)", 90, 200, 120)
ap_lo = st.sidebar.slider("Diastolic BP (ap_lo)", 60, 150, 80)
cholesterol = st.sidebar.selectbox("Cholesterol", [1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1])
glucose = st.sidebar.selectbox("Glucose", [1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1])
smoke = st.sidebar.selectbox("Smoking", [0, 1], format_func=lambda x: "Yes" if x else "No")
alco = st.sidebar.selectbox("Alcohol Intake", [0, 1], format_func=lambda x: "Yes" if x else "No")
active = st.sidebar.selectbox("Physical Activity", [0, 1], format_func=lambda x: "Yes" if x else "No")
gender = st.sidebar.selectbox("Gender", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")

# ------------------------------
# Feature Engineering
# ------------------------------
height_cm = height_feet * 30.48  # Convert feet to cm
age_days = age_years * 365
bmi = weight / ((height_cm / 100) ** 2)

# ------------------------------
# Prepare Input DataFrame
# ------------------------------
user_input = pd.DataFrame({
    'age': [age_days],
    'gender': [gender],
    'height': [height_cm],
    'weight': [weight],
    'ap_hi': [ap_hi],
    'ap_lo': [ap_lo],
    'cholesterol': [cholesterol],
    'gluc': [glucose],
    'smoke': [smoke],
    'alco': [alco],
    'active': [active]
})

# Reorder columns to match scaler's training
user_input = user_input[scaler.feature_names_in_]

# ------------------------------
# Model Prediction
# ------------------------------
X_scaled = scaler.transform(user_input)
pred = model.predict(X_scaled)[0]
pred_prob = model.predict_proba(X_scaled)[0][1]

# ------------------------------
# Display Prediction
# ------------------------------
st.subheader("Prediction")

import plotly.graph_objects as go

# Display textual message
if pred == 1:
    st.error(f"⚠️ High Risk of Cardiovascular Disease ({pred_prob * 100:.2f}%)")
else:
    st.success(f"✅ Low Risk of Cardiovascular Disease ({(1 - pred_prob) * 100:.2f}%)")

# Gauge Chart for Risk Probability
gauge_fig = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=pred_prob * 100,
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': "Cardiovascular Risk (%)", 'font': {'size': 24}},
    delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
    gauge={
        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 40], 'color': '#5cb85c'},      # Green
            {'range': [40, 70], 'color': '#f0ad4e'},     # Orange
            {'range': [70, 100], 'color': '#d9534f'}     # Red
        ],
        'threshold': {
            'line': {'color': "black", 'width': 4},
            'thickness': 0.75,
            'value': pred_prob * 100
        }
    }
))

st.plotly_chart(gauge_fig, use_container_width=True)



# ------------------------------
# EDA Section
# ------------------------------
st.subheader("Exploratory Data Analysis")

# Add BMI if missing
if 'BMI' not in df.columns:
    df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(px.histogram(df, x="age", color="cardio", nbins=50, title="Age Distribution by Cardio Outcome"), use_container_width=True)
    st.plotly_chart(px.scatter(df, x="weight", y="ap_lo", color="cardio", title="Weight vs Diastolic BP"), use_container_width=True)

with col2:
    st.plotly_chart(px.histogram(df, x="BMI", color="cardio", nbins=50, title="BMI Distribution by Cardio Outcome"), use_container_width=True)
    st.plotly_chart(px.scatter(df, x="age", y="ap_hi", color="cardio", title="Age vs Systolic BP"), use_container_width=True)

st.plotly_chart(px.histogram(df, x="cholesterol", color="cardio", barmode="group", title="Cholesterol Levels by Cardio Outcome"), use_container_width=True)
st.plotly_chart(px.histogram(df, x="gluc", color="cardio", barmode="group", title="Glucose Levels by Cardio Outcome"), use_container_width=True)

# ------------------------------
# Conclusion
# ------------------------------
st.subheader("Conclusion")
st.markdown("""
- This app enables users to estimate their cardiovascular disease risk based on health and lifestyle inputs.
- Logistic Regression is used due to its interpretability and simplicity.
- The visualizations help you explore how factors like age, BMI, and blood pressure correlate with heart disease outcomes.
""")
st.markdown("<hr><center>Developed by Zarwa | IDS Project 2025</center>", unsafe_allow_html=True)
