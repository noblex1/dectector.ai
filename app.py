
import streamlit as st
import pandas as pd
import joblib
import os

# Auto-train if model files are missing
if not os.path.exists("model.pkl") or not os.path.exists("scaler.pkl"):
    from train_model import train_and_save_model
    train_and_save_model()

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("AI Intrusion Detection System")

uploaded_file = st.file_uploader("Upload CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data preview:", df.head())

    numeric_data = df.select_dtypes(include=['int64', 'float64'])
    scaled_data = scaler.transform(numeric_data)
    preds = model.predict(scaled_data)
    df["Prediction"] = ["Intrusion" if x == -1 else "Normal" for x in preds]

    st.write("Prediction Results:", df)
