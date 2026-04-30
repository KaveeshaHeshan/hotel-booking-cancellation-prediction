# 🏨 Hotel Booking Cancellation Prediction

> Predicting hotel booking cancellations using Logistic Regression

---

## 📌 Project Overview

A machine learning model that predicts whether a hotel booking will be cancelled or not. Built using Logistic Regression on the Hotel Booking Demand dataset containing 119,390 real hotel booking records. The project includes a complete end-to-end ML pipeline covering data preprocessing, SMOTE class balancing, feature engineering, model training and evaluation. A Streamlit web application is included for real-time cancellation prediction.

---

## 📁 Dataset

| Detail | Value |
|--------|-------|
| Source | [Hotel Booking Demand — Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) |
| Records | 119,390 hotel bookings |
| Features | 32 columns |
| Target | `is_canceled` (0 = Not Cancelled, 1 = Cancelled) |

---

## 🤖 Algorithm

**Logistic Regression** (Binary Classification)

---

## ⚙️ Preprocessing Steps

1. Duplicate removal (31,994 records removed)
2. Missing value treatment
3. Data leakage column removal
4. Outlier capping (adr column)
5. Label Encoding (10 columns)
6. SMOTE oversampling (72.5% to 50/50)
7. Feature Engineering (27 to 31 features)
8. StandardScaler normalisation

---

## 📈 Results

| Metric | Score |
|--------|-------|
| Accuracy | 71.21% |
| Precision | 48.50% |
| Recall | 76.27% |
| F1 Score | 59.29% |
| AUC-ROC | 80.31% |
| Train-Test Gap | 2.31% |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.x |
| ML Library | scikit-learn |
| Data Processing | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Class Balancing | imbalanced-learn (SMOTE) |
| Web App | Streamlit |
| Model Saving | pickle |

---

## 🌐 Web Application

Built with **Streamlit** for real-time cancellation prediction.

```bash
streamlit run app.py




<div align="center">
  Hotel Booking Cancellation Prediction · Logistic Regression · Python · Streamlit
</div>
