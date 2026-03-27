import pandas as pd
import os

# =====================================
# PROJECT PATH
# =====================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(
    BASE_DIR,
    "docs",
    "data",
    "soil_moisture_dataset_real.csv"
)

# =====================================
# LOAD DATASET
# =====================================
data = pd.read_csv(data_path)

print("✅ Dataset Loaded:", data.shape)

# =====================================
# REMOVE USELESS GEE COLUMNS ✅
# =====================================
data = data.drop(
    columns=["system:index", ".geo"],
    errors="ignore"
)

print("✅ Removed unnecessary columns")

# =====================================
# REMOVE MISSING VALUES
# =====================================
data = data.dropna()

# =====================================
# FEATURE ENGINEERING ⭐⭐⭐
# =====================================

# SAR Ratio
data["VV_VH_ratio"] = data["VV"] / (data["VH"] + 1e-6)

# Basic SAR relations
data["VV_minus_VH"] = data["VV"] - data["VH"]
data["VV_plus_VH"] = data["VV"] + data["VH"]

# Vegetation interaction
data["NDVI_VV"] = data["NDVI"] * data["VV"]
data["NDVI_VH"] = data["NDVI"] * data["VH"]

# Non-linear radar response
data["VV_squared"] = data["VV"] ** 2
data["VH_squared"] = data["VH"] ** 2
data["NDVI_squared"] = data["NDVI"] ** 2

# ⭐ SAR PHYSICS FEATURES (BIG BOOST)
data["SAR_energy"] = (data["VV"]**2 + data["VH"]**2)
data["SAR_product"] = data["VV"] * data["VH"]
data["NDVI_ratio"] = data["NDVI"] / (data["VV_VH_ratio"] + 1e-6)

print("✅ Feature Engineering Completed")

# =====================================
# SAVE PROCESSED DATA
# =====================================
processed_path = os.path.join(
    BASE_DIR,
    "docs",
    "data",
    "processed_data.csv"
)

data.to_csv(processed_path, index=False)

print("✅ Processed Dataset Saved Successfully")
print("✅ Final Dataset Shape:", data.shape)