# hotel-booking-cancellation-prediction
Predicting hotel booking cancellations using 
Logistic Regression

Project Overview
This project predicts whether a hotel booking 
will be cancelled using machine learning. 
Built as part of IT4060 Machine Learning 
Assignment — Member 1.

Dataset
- Source   : Hotel Booking Demand (Kaggle)
- Records  : 119,390 hotel bookings
- Features : 32 columns
- Target   : is_canceled (0 or 1)

Algorithm
Logistic Regression (Binary Classification)

Preprocessing
- Duplicate removal (31,994 records removed)
- Missing value treatment
- Data leakage column removal
- Outlier capping (adr column)
- Label Encoding (10 columns)
- SMOTE oversampling (72.5% to 50/50)
- Feature Engineering (27 to 31 features)
- StandardScaler normalisation

Results
| Metric    | Score  |
|-----------|--------|
| Accuracy  | 71.21% |
| Precision | 48.50% |
| Recall    | 76.27% |
| F1 Score  | 59.29% |
| AUC-ROC   | 80.31% |
| Gap       |  2.31% |

Tech Stack
Python, scikit-learn, pandas, numpy,
matplotlib, seaborn, imbalanced-learn,
streamlit, pickle

Web App
Streamlit web application included.
Run: streamlit run app.py
URL: http://localhost:8501
