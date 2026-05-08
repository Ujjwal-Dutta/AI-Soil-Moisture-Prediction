import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Soil Moisture Prediction",
    layout="wide"
)

# -----------------------------
# Base Directory
# -----------------------------
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# -----------------------------
# File Paths
# -----------------------------
model_path = os.path.join(
    BASE_DIR,
    "docs",
    "best_model.pkl"
)

feature_path = os.path.join(
    BASE_DIR,
    "docs",
    "feature_columns.pkl"
)

data_path = os.path.join(
    BASE_DIR,
    "docs",
    "data",
    "processed_data.csv"
)

# -----------------------------
# Load Model + Features
# -----------------------------
try:

    model = joblib.load(model_path)

    selected_features = joblib.load(
        feature_path
    )

except Exception as e:

    st.error(
        f"Error loading model files: {e}"
    )

    st.stop()

# -----------------------------
# Title
# -----------------------------
st.title(
    "🌱 AI-Based Soil Moisture Prediction System"
)

st.write("""
This dashboard predicts soil moisture using
Sentinel-1 SAR and Sentinel-2 satellite data
with Machine Learning techniques.
""")

# -----------------------------
# Load Dataset
# -----------------------------
try:

    data = pd.read_csv(data_path)

except Exception as e:

    st.error(
        f"Error loading dataset: {e}"
    )

    st.stop()

# -----------------------------
# Keep Numeric Columns
# -----------------------------
data = data.select_dtypes(
    include=["number"]
)

# -----------------------------
# Dataset Preview
# -----------------------------
st.subheader("📊 Dataset Preview")

st.dataframe(data.head())

# -----------------------------
# Validate Features
# -----------------------------
missing_cols = [
    col for col in selected_features
    if col not in data.columns
]

if missing_cols:

    st.error(
        f"Dataset missing required features: "
        f"{missing_cols}"
    )

    st.stop()

# -----------------------------
# User Sample Selection
# -----------------------------
st.subheader(
    "🎛 Select Input Sample"
)

row_id = st.slider(
    "Select Sample Index",
    min_value=0,
    max_value=len(data) - 1,
    value=0
)

# -----------------------------
# Prepare Input
# -----------------------------
sample = data[selected_features].iloc[[row_id]]

sample = sample.reindex(
    columns=selected_features
)

# -----------------------------
# Display Features
# -----------------------------
st.subheader(
    "🛰 Selected Input Features"
)

st.dataframe(sample)

# -----------------------------
# Prediction
# -----------------------------
try:

    prediction = model.predict(sample)

    st.subheader(
        "🌍 Predicted Soil Moisture"
    )

    st.success(
        f"Soil Moisture Value: "
        f"{prediction[0]:.4f}"
    )

except Exception as e:

    st.error(
        f"Prediction failed: {e}"
    )

# -----------------------------
# Charts
# -----------------------------
st.subheader(
    "📈 NDVI Distribution"
)

if "NDVI" in data.columns:

    st.bar_chart(data["NDVI"])

else:

    st.warning(
        "NDVI column not found."
    )

st.subheader(
    "📉 VV Backscatter Distribution"
)

if "VV" in data.columns:

    st.line_chart(data["VV"])

else:

    st.warning(
        "VV column not found."
    )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.markdown("""
Developed using Sentinel-1 SAR,
Sentinel-2 NDVI, and Machine Learning.
""")
