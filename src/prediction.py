import pandas as pd
import os
import joblib

# -----------------------------
# Project Paths
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

print("✅ Model & Features Loaded Successfully")

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv(data_path)

# Keep numeric columns
data = data.select_dtypes(include=["number"])

# -----------------------------
# Prepare Sample Input
# -----------------------------
print("\n🔍 Selected Features:", selected_features)

# Select correct features
sample = data[selected_features].iloc[0:5]

# FIX 1: Handle missing values
sample = sample.fillna(0)

print("\n📥 Sample Input:")
print(sample)

# FIX 2: Check shape
if sample.shape[1] != len(selected_features):
    raise ValueError("❌ Feature mismatch! Check selected_features")

# -----------------------------
# Prediction
# -----------------------------
predictions = model.predict(sample)

print("\n🌱 Predicted Soil Moisture:")
print(predictions)

# -----------------------------
# Save Prediction Results
# -----------------------------
output = sample.copy()

# FIX 3: Ensure predictions are clean numbers
output["Predicted_Soil_Moisture"] = predictions.flatten()

output_path = os.path.join(BASE_DIR, "docs", "predicted_results.csv")

output.to_csv(output_path, index=False)

print("\n✅ Results Saved Successfully!")
print("📁 File Location:", output_path)