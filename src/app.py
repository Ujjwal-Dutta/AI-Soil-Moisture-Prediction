import streamlit as st
import pandas as pd
import joblib
import os
import gdown

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
# Create Required Directories
# -----------------------------
os.makedirs(
    os.path.dirname(model_path),
    exist_ok=True
)

os.makedirs(
    os.path.dirname(feature_path),
    exist_ok=True
)

os.makedirs(
    os.path.dirname(data_path),
    exist_ok=True
)

# -----------------------------
# Google Drive Share Links
# -----------------------------

# best_model.pkl (NEW LINK)
MODEL_URL = (
    "https://drive.google.com/file/d/"
    "1BlzKNSX2WPsYsISQAYau0CLuBoPSa-MF/view?usp=drive_link"
)

# feature_columns.pkl
FEATURE_URL = (
    "https://drive.google.com/file/d/"
    "11k67yTAlSnZ8gOyy_MwV9vdlt83Dqq1M/view?usp=sharing"
)

# processed_data.csv
DATA_URL = (
    "https://drive.google.com/file/d/"
    "13ttpzkJ6i1X5RMJpIx1_k8ejxbQ7xfVh/view?usp=sharing"
)

URLS = {
    model_path: MODEL_URL,
    feature_path: FEATURE_URL,
    data_path: DATA_URL
}

# -----------------------------
# Download Files If Missing
# -----------------------------
for path, url in URLS.items():

    if not os.path.exists(path):

        with st.spinner(
            f"Downloading {os.path.basename(path)}..."
        ):

            try:

                gdown.cached_download(
                    url=url,
                    path=path,
                    quiet=False,
                    fuzzy=True
                )

            except Exception as e:

                st.error(
                    f"Failed to download "
                    f"{os.path.basename(path)}: {e}"
                )

                st.stop()

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
# App Title
# -----------------------------
st.title(
    "🌱 AI-Based Soil Moisture Prediction System"
)

st.write("""
This dashboard predicts soil moisture using
Sentinel-1 SAR and Sentinel-2 satellite data
with Machine Learning.
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
# Keep Numeric Columns Only
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
# Feature Validation
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
# Display Input Features
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
# Visualizations
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
