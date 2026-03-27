import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# =====================================
# PROJECT ROOT PATH ✅
# =====================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(
    BASE_DIR,
    "docs",
    "data",
    "processed_data.csv"
)

model_dir = os.path.join(BASE_DIR, "docs")
results_dir = os.path.join(BASE_DIR, "docs")

os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# =====================================
# LOAD DATA
# =====================================
df = pd.read_csv(data_path)

print("✅ Dataset Loaded:", df.shape)

target = "Soil_Moisture"

if target not in df.columns:
    raise ValueError("❌ Soil_Moisture column missing")

X = df.drop(columns=[target])
y = df[target]

print("✅ Features:", list(X.columns))
# =====================================
# TRAIN TEST SPLIT
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("✅ Data Split Completed")
# =====================================
# FEATURE SELECTION (ADD HERE 🔥)
# =====================================
selector_model = RandomForestRegressor(n_estimators=300, random_state=42)
selector_model.fit(X_train, y_train)

importance = pd.Series(selector_model.feature_importances_, index=X.columns)

top_features = importance.sort_values(ascending=False).head(8).index.tolist()

print("🔥 Selected Features:", top_features)

# Apply to train and test
X_train = X_train[top_features]
X_test = X_test[top_features]
X = X[top_features]

# =====================================
# MODELS
# =====================================
models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=600,
    max_depth=18,
    min_samples_split=4,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    random_state=42,
    n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=800,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=2
    )
}

best_model = None
best_score = -999

# =====================================
# TRAIN + EVALUATION
# =====================================
for name, model in models.items():

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    cv_score = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="r2",
    n_jobs=1   # ✅ MEMORY SAFE
).mean()

    print(f"\n{name}")
    print("R2 :", round(r2,4))
    print("RMSE :", round(rmse,4))
    print("CV R2 :", round(cv_score,4))

    if r2 > best_score:
        best_score = r2
        best_model = model

print("\n🏆 Best Model Selected:", type(best_model).__name__)

# =====================================
# SAVE MODEL
# =====================================
model_path = os.path.join(model_dir, "best_model.pkl")
joblib.dump(best_model, model_path)

print("✅ Model Saved Successfully")
# =====================================
# SAVE SELECTED FEATURES (ADD THIS 🔥)
# =====================================
feature_path = os.path.join(model_dir, "feature_columns.pkl")
joblib.dump(top_features, feature_path)

print("✅ Feature Columns Saved Successfully")

# =====================================
# FEATURE IMPORTANCE
# =====================================
importance = best_model.feature_importances_

plt.figure(figsize=(8,6))

pd.Series(
    importance,
    index=X.columns
).sort_values().plot(kind="barh")

plt.title("Feature Importance")
plt.tight_layout()

plot_path = os.path.join(results_dir, "feature_importance.png")
plt.savefig(plot_path)

print("✅ Feature Importance Saved")