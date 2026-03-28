
# 🌱 AI-Based Soil Moisture Prediction Using Multi-Sensor Satellite Data

## 📌 Project Overview
This project implements a multi-sensor data fusion approach using Sentinel-1 (SAR) and Sentinel-2 (Optical) satellite data to predict soil moisture using machine learning models.

The system leverages advanced regression techniques to provide accurate soil moisture estimation, supporting precision agriculture, drought monitoring, and water resource management.

---

## 🚀 Live Demo
👉 Try the deployed application here:  
https://ai-soil-moisture-prediction-4yx997d2xsmfkhtfctinxw.streamlit.app/

---

## 🧠 Features
- 📡 Multi-sensor satellite data fusion (Sentinel-1 + Sentinel-2)  
- 🤖 Machine Learning-based prediction  
- 📊 Interactive Streamlit dashboard  
- 📈 Visualization of satellite-derived features  
- ⚡ Real-time inference using trained models  
- ☁️ Cloud-deployed web application  

---

## 🛠️ Technology Stack
- Google Earth Engine  
- Python  
- Pandas & NumPy  
- Scikit-learn  
- XGBoost  
- Streamlit  
- Joblib  

---

## 🤖 Models Used
- 🌳 Random Forest Regression  
- ⚡ XGBoost Regression  

---

## 📊 Evaluation Metrics
- RMSE (Root Mean Squared Error)  
- MAE (Mean Absolute Error)  
- R² Score  

---

## ☁️ Data & Model Storage (Important)

⚠️ This repository does NOT include the `docs/` folder due to size limitations.

Instead, all required files are hosted on Google Drive and automatically downloaded in the Streamlit app using `gdown`.

### 📦 Required Files (Google Drive Links)

🧠 **Trained Model (`best_model.pkl`)**  
https://drive.google.com/file/d/1xP2sVtuFsR8OIj3ByDKcL3hYeLN4rVZf/view?usp=sharing  

📑 **Feature Columns (`feature_columns.pkl`)**  
https://drive.google.com/file/d/11k67yTAlSnZ8gOyy_MwV9vdlt83Dqq1M/view?usp=sharing  

📊 **Processed Dataset (`processed_data.csv`)**  
https://drive.google.com/file/d/13ttpzkJ6i1X5RMJpIx1_k8ejxbQ7xfVh/view?usp=sharing  

---

## 📁 Project Structure
AI-Soil-Moisture-Prediction/
│
├── src/
│ └── app.py
├── requirements.txt
└── README.md


---

## ⚙️ How It Works
- Satellite data from Sentinel-1 and Sentinel-2 is processed using Google Earth Engine  
- Features like NDVI, VV, VH are extracted  
- Machine learning models are trained on processed data  
- The Streamlit app loads the model and features from Google Drive  
- User selects a sample → model predicts soil moisture  

---

## 🎯 Applications
- 🌾 Precision Agriculture  
- 🌍 Drought Monitoring  
- 💧 Water Resource Management  
- 🌱 Crop Health Analysis  

---

## ▶️ Run Locally
git clone https://github.com/your-username/ai-soil-moisture-prediction.git
cd ai-soil-moisture-prediction
pip install -r requirements.txt
cd src
streamlit run app.py


---

## ☁️ Deployment
- Hosted on Streamlit Community Cloud  
- Automatically fetches required files from Google Drive  
- Updates reflect instantly after GitHub commits  

---

## 📌 Future Improvements
- Real-time satellite data integration  
- API-based backend (FastAPI/Flask)  
- Improved UI/UX dashboard  
- Model optimization and ensemble methods  
- Additional satellite data sources  
