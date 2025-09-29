import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the trained model
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Tank Runout Time Predictor", layout="wide")

st.title("⛽ Petrol Tank Runout Time Predictor")
st.markdown("""
This app predicts the **time (hours/days) until a petrol tank runs out** 
based on current sales, tank level, and environmental factors.
""")

# --- User Inputs ---
st.sidebar.header("Input Features")

def user_input_features():
    year = st.sidebar.number_input("Year", min_value=2000, max_value=2100, value=2025)
    month = st.sidebar.number_input("Month", min_value=1, max_value=12, value=1)
    day = st.sidebar.number_input("Day", min_value=1, max_value=31, value=1)
    petrol_liters_sold = st.sidebar.number_input("Petrol Liters Sold", min_value=0.0, value=500.0)
    petrol_price_per_liter = st.sidebar.number_input("Petrol Price per Liter", min_value=0.0, value=1.2)
    temperature = st.sidebar.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, value=25.0)
    petrol_total_sales = st.sidebar.number_input("Petrol Total Sales", min_value=0.0, value=600.0)
    tank_level = st.sidebar.number_input("Tank Level (Liters)", min_value=0.0, value=2000.0)

    data = {
        'Year': year,
        'Month': month,
        'Day': day,
        'Petrol_Liters_Sold': petrol_liters_sold,
        'Petrol_Price_Per_Liter': petrol_price_per_liter,
        'Temperature': temperature,
        'Petrol_Total_Sales': petrol_total_sales,
        'Tank_Level': tank_level
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader("Input Features")
st.write(input_df)

# --- Prediction ---
prediction = model.predict(input_df)[0]

st.subheader("⛽ Predicted Time to Run Out Tank")
st.success(f"{prediction:.2f} hours")  # or days depending on your model

# Optional: Add chart showing predicted consumption vs tank level
st.subheader("Tank Level vs Expected Consumption")
st.bar_chart({
    "Tank Level": [input_df['Tank_Level'][0]],
    "Expected Consumption per Hour": [input_df['Petrol_Liters_Sold'][0]]
})

