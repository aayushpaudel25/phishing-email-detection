# Phishing Email Detection (Random Forest & XGBoost)

## 1. Project Overview

This project builds a **phishing email detection** system using two machine learning models:

- **Random Forest** (Phase 1)
- **XGBoost** (Phase 2)

Emails are represented using **TF-IDF** features extracted from the email text.  
The goal is to compare both models using metrics like **accuracy, precision, recall, F1-score, and confusion matrix**.

---

## 2. Project Structure

```text
phishing_email_detection/
├── data/
│   ├── raw/          
│   └── processed/    
├── models/           
├── notebooks/        
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── utils.py
│   ├── train_random_forest.py
│   ├── train_xgboost.py
│   ├── evaluate_models.py
│   └── main.py
├── requirements.txt
└── README.md
