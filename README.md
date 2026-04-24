# 🏨 Hotel Booking Cancellation Prediction

> Predicting hotel booking cancellations using Logistic Regression

---

## 📊 Project Overview
This project predicts whether a hotel booking
will be cancelled using machine learning.
Built as part of IT4060 Machine Learning
Assignment — Member 1.

---

## 📁 Dataset
| Detail | Value |
|--------|-------|
| Source | Hotel Booking Demand (Kaggle) |
| Records | 119,390 hotel bookings |
| Features | 32 columns |
| Target | is_canceled (0 or 1) |

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
