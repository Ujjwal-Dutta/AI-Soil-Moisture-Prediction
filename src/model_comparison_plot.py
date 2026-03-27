import matplotlib.pyplot as plt
import os

# =============================
# Project Path
# =============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_path = os.path.join(BASE_DIR, "docs")
os.makedirs(results_path, exist_ok=True)

# =============================
# Model Names & Scores
# =============================
models = ["Random Forest", "XGBoost"]
r2_scores = [0.5177, 0.4758]  # Use latest values from training output

# =============================
# Plot
# =============================
plt.figure(figsize=(6,4))
plt.bar(models, r2_scores, color=["skyblue", "salmon"])
plt.ylabel("R² Score")
plt.title("Model Performance Comparison")
plt.ylim(0,1)  # R² max 1 for clarity
plt.tight_layout()

# =============================
# Save Plot
# =============================
plot_file = os.path.join(results_path, "model_comparison.png")
plt.savefig(plot_file)

print("✅ Model comparison plot saved at:", plot_file)