import streamlit as st
import pandas as pd
import joblib
import os
import gdown

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "docs", "best_model.pkl")
feature_path = os.path.join(BASE_DIR, "docs", "feature_columns.pkl")
data_path = os.path.join(BASE_DIR, "docs", "data", "processed_data.csv")

# -----------------------------
# Google Drive File IDs
# -----------------------------
FILES = {
    model_path: "1cr9VJ_ofMxbJpLcTwMsbSj6_dtNfA2OX",   # soil moisture model
    feature_path: "11k67yTAlSnZ8gOyy_MwV9vdlt83Dqq1M"  # feature columns
}

# -----------------------------
# Download files if missing
# -----------------------------
os.makedirs(os.path.join(BASE_DIR, "docs"), exist_ok=True)

for path, file_id in FILES.items():
    if not os.path.exists(path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, path, quiet=False)

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
data = data.select_dtypes(include=["number"])

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
