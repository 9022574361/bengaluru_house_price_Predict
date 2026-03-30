# 🏠 Bengaluru House Price Prediction using XGBoost

## 📌 Project Overview

This project is a **Machine Learning–powered web application** that predicts house prices in Bengaluru based on important property features.
The system uses an optimized **XGBoost Regression model** trained on cleaned and engineered real-estate data to provide accurate price estimations.

The application allows users to input property details and instantly receive an estimated house price through an interactive **Streamlit web interface**.

---

## 🚀 Key Features

* ✅ End-to-end Machine Learning pipeline
* ✅ Advanced feature engineering & outlier removal
* ✅ XGBoost regression for high prediction accuracy
* ✅ Scaled feature preprocessing using StandardScaler
* ✅ Interactive Streamlit UI
* ✅ Real-time house price prediction
* ✅ Production-ready model serialization using Pickle

---

## 🧠 Machine Learning Workflow

1. Data Cleaning & Preprocessing
2. Feature Engineering
3. Outlier Detection & Removal
4. Feature Selection
5. Train/Test Split
6. Feature Scaling (StandardScaler)
7. Model Training using XGBoost Regressor
8. Model Evaluation (R² Score & MAE)
9. Model Deployment with Streamlit

---

## 📊 Selected Features

The model predicts price based on:

* `total_sqft` — Total property area
* `balcony` — Number of balconies
* `bedroom_count` — Total bedrooms
* `area_sqft` — Usable area
* `bath_to_room_ratio` — Bathroom to room ratio

---

## ⚙️ Model Performance

Evaluation metrics used:

* **R² Score** → Measures prediction accuracy
* **Mean Absolute Error (MAE)** → Average prediction error

The XGBoost model provides strong generalization and stable predictions on unseen data.

---

## 🧩 Tech Stack

* Python
* Pandas & NumPy
* Scikit-learn
* XGBoost
* Streamlit
* Pickle (Model Serialization)

---

## 📁 Project Structure

```
project/
│
├── app.py                     # Streamlit web application
├── train_model.py             # Model training pipeline
├── bengaluru_house_model.pkl  # Saved model + scaler
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Train the Model

```bash
python train_model.py
```

### 3️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---

## 🌐 Application Demo

Users can enter property details and instantly receive an AI-generated house price prediction.

---

## 📈 Future Improvements

* Location-based prediction enhancement
* Hyperparameter optimization
* Model explainability (SHAP values)
* Cloud deployment (Render / Railway)
* Real-time data integration

---

## 👨‍💻 Author

**Abhi**

Machine Learning Enthusiast | Data Science Projects | AI Applications

---

## ⭐ Acknowledgements

This project demonstrates a complete real-world ML lifecycle — from data preprocessing to deployment — and is intended for learning, experimentation, and portfolio demonstration.

