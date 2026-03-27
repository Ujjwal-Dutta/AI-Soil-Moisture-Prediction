import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# =============================
# PROJECT BASE PATH
# =============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =============================
# PATHS
# =============================
data_path = os.path.join(
    BASE_DIR,
    "docs",
    "data",
    "processed_data.csv"
)

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

results_path = os.path.join(BASE_DIR, "docs")
os.makedirs(results_path, exist_ok=True)

# =============================
# LOAD DATA
# =============================
df = pd.read_csv(data_path)

print("✅ Dataset Loaded:", df.shape)

df = df.select_dtypes(include=["number"])

# =============================
# FEATURES & TARGET
# =============================
X = df.drop(columns=["Soil_Moisture"])
y = df["Soil_Moisture"]

# =============================
# SAME SPLIT
# =============================
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("✅ Test Data Prepared")

# =============================
# LOAD MODEL + FEATURES
# =============================
model = joblib.load(model_path)
selected_features = joblib.load(feature_path)

print("✅ Model & Features Loaded")

# 🔥 IMPORTANT: USE SAME FEATURES
X_test = X_test[selected_features]

# =============================
# PREDICTION
# =============================
preds = model.predict(X_test)

# =============================
# METRICS
# =============================
r2 = r2_score(y_test, preds)
rmse = mean_squared_error(y_test, preds) ** 0.5

print("\n📊 Evaluation Results (TEST DATA)")
print("R2 :", round(r2, 4))
print("RMSE :", round(rmse, 4))
print("Approx Accuracy (%):", round(r2 * 100, 2))

# =============================
# ACTUAL vs PREDICTED PLOT
# =============================
plt.figure(figsize=(6,6))

plt.scatter(y_test, preds, alpha=0.6)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--"
)

plt.xlabel("Actual Soil Moisture")
plt.ylabel("Predicted Soil Moisture")
plt.title("Actual vs Predicted Soil Moisture")

plt.tight_layout()

plot_path = os.path.join(
    results_path,
    "actual_vs_predicted.png"
)

plt.savefig(plot_path)

print("✅ Plot Saved inside docs/")