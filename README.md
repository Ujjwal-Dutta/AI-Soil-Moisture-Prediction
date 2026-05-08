# 🌱 AI-Based Soil Moisture Prediction Using Multi-Sensor Satellite Data

## 📌 Project Overview
This project implements a Machine Learning based soil moisture prediction system using multi-sensor satellite data from:

- Sentinel-1 SAR Data
- Sentinel-2 Optical Data

The system predicts soil moisture levels using extracted satellite features and a trained machine learning model.

---

# 🚀 Live Streamlit Dashboard

## 🌐 Dashboard Link

https://ai-soil-moisture-prediction-4yx997d2xsmfkhtfctinxw.streamlit.app/

---

# 🛰️ Features Used

The model uses important satellite-derived features such as:

- NDVI
- VV Backscatter
- VH Backscatter
- Radar Features
- Optical Features
- Environmental Parameters

---

# 🤖 Machine Learning Model

The trained ML model predicts soil moisture values using selected satellite features.

### ✅ Best Performing Model
- Random Forest Regressor

### 📊 Model Performance
- R² Score: 0.89
- MAE: Low Error
- RMSE: Optimized Performance

---

# 📂 Project Structure

```bash
AI-Soil-Moisture-Prediction/
│
├── docs/
│   ├── best_model.pkl
│   ├── feature_columns.pkl
│   └── data/
│       └── processed_data.csv
│
├── src/
│   └── app.py
│
├── requirements.txt
├── README.md
```

---

# 📦 Required Files (Google Drive Links)

## 🧠 Trained Model (best_model.pkl)
https://drive.google.com/file/d/1BlzKNSX2WPsYsISQAYau0CLuBoPSa-MF/view?usp=drive_link

## 📑 Feature Columns (feature_columns.pkl)
https://drive.google.com/file/d/11k67yTAlSnZ8gOyy_MwV9vdlt83Dqq1M/view?usp=sharing

## 📊 Processed Dataset (processed_data.csv)
https://drive.google.com/file/d/13ttpzkJ6i1X5RMJpIx1_k8ejxbQ7xfVh/view?usp=sharing

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/Ujjwal-Dutta/AI-Soil-Moisture-Prediction.git
```

## 2️⃣ Move Into Project Folder

```bash
cd AI-Soil-Moisture-Prediction
```

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Streamlit App

```bash
streamlit run src/app.py
```

---

# 📋 requirements.txt

```text
streamlit
pandas
joblib
scikit-learn
gdown
```

---

# 📈 Dashboard Features

✅ Satellite dataset preview  
✅ Soil moisture prediction  
✅ Interactive sample selection  
✅ NDVI visualization  
✅ VV backscatter visualization  
✅ Machine learning inference  

---

# 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- Joblib
- Google Drive API Downloading
- Sentinel-1 SAR Data
- Sentinel-2 Optical Data

---

# 🌍 Application Areas

- Precision Agriculture
- Smart Irrigation
- Drought Monitoring
- Agricultural Analytics
- Remote Sensing Research

---

# 👨‍💻 Developed By

Ujjwal Dutta

---

# 📜 License

This project is developed for educational and research purposes.
