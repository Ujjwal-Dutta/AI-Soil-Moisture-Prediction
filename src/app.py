import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "docs", "best_model.pkl")
feature_path = os.path.join(BASE_DIR, "docs", "feature_columns.pkl")
data_path = os.path.join(BASE_DIR, "docs", "data", "processed_data.csv")

# -----------------------------
# Load Model + Features
# -----------------------------
model = joblib.load(model_path)
selected_features = joblib.load(feature_path)

st.title("🌱 AI-Based Soil Moisture Prediction System")

st.write("""
This dashboard predicts soil moisture using
Sentinel-1 SAR and Sentinel-2 NDVI satellite data.
""")

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv(data_path)
data = data.select_dtypes(include=["number"])  # Safety

st.subheader("📊 Satellite Dataset Preview")
st.dataframe(data.head())

# -----------------------------
# User Selection
# -----------------------------
row_id = st.slider(
    "Select Sample Index",
    0,
    len(data)-1,
    0
)

# Use only selected features
sample = data.loc[[row_id], selected_features]

st.subheader("Selected Input Features")
st.dataframe(sample)

# -----------------------------
# Prediction
# -----------------------------
prediction = model.predict(sample)

st.subheader("🌍 Predicted Soil Moisture")
st.success(f"Soil Moisture Value: {prediction[0]:.3f}")

# -----------------------------
# Visualization
# -----------------------------
st.subheader("NDVI Distribution")
st.bar_chart(data["NDVI"])

st.subheader("VV Backscatter Distribution")
st.line_chart(data["VV"])