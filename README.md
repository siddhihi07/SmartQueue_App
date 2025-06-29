# SmartQueue: AI-Based Patient Triage System

SmartQueue is an AI-powered triage system developed as part of a Machine Learning Internship at **Feynn Labs**. It assists hospitals and clinics in prioritizing OPD patients based on symptom severity. Built using Python, Streamlit, and a trained Random Forest Classifier, it helps healthcare providers classify patients into Low, Medium, or High risk — in real time.

---

## Features

- ✅ Streamlit web interface for real-time triage
- ✅ Random Forest ML model for risk prediction
- ✅ Dummy patient generator for testing
- ✅ Color-coded alerts for critical cases
- ✅ Risk category bar chart (optional)
- ✅ Export patient queue data to CSV

---

## Why SmartQueue?

- Reduces patient waiting time
- Ensures high-risk cases are prioritized
- Ideal for Tier-2/3 hospitals with minimal HMS infrastructure
- Lightweight & deployable on local or cloud servers
- Built with simplicity and scalability in mind

---

## Tech Stack

- **Frontend/UI**: Streamlit
- **ML Model**: Random Forest Classifier (scikit-learn)
- **Backend**: Python 3.12
- **Data Handling**: pandas, NumPy
- **Model Serialization**: joblib

---
## Sample Patient Prediction
You can generate a dummy patient in the app or manually input symptoms.
The model predicts a risk score and maps it to a label:

🟢 Low Risk

🟡 Medium Risk

🔴 High Risk

---

## Financial Forecast (from report)
Revenue = (Monthly Fee × Hospitals) - Operating Costs

Sample:
Revenue = (100 × ₹1999) - ₹8000 = ₹1,91,900/month
