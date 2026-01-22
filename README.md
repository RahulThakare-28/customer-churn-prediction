# Customer Churn Prediction

## Overview

A modular, production-ready **Customer Churn Prediction system** built using machine learning, designed for easy experimentation, evaluation, and deployment as a web application.

The project emphasizes **clean architecture**, **leakage-safe preprocessing**, and **model reproducibility**, making it suitable for real-world ML workflows.

---

## Project Goals

* Predict customer churn with high reliability
* Build a **modular & loosely coupled ML pipeline**
* Ensure **data leakage prevention**
* Support seamless **model comparison and deployment**
* Integrate with a **web application backend**

---

## Current Status 

### Data & Preprocessing

* Data cleaning and validation completed
* Custom preprocessing pipeline implemented
* Leakage-safe feature engineering
* Pipeline serialized using `joblib`

### Modeling

* Models trained and evaluated:

  * Logistic Regression (baseline)
  * Random Forest
  * AdaBoost
* Model evaluation using:

  * Accuracy
  * Recall
  * ROC-AUC
  * Confusion Matrix
* Trained models stored as versioned artifacts
* GitHub release: **v1.0-modeling-complete**

### Backend & API

* MongoDB setup completed
* FastAPI backend integration in progress
* Models and preprocessing pipeline ready for inference

---

## Project Structure

```
customer-churn-prediction/
│
├── artifacts/
│   ├── preprocessing.joblib
│   └── selected_models/
│       ├── logistic_model.joblib
│       ├── random_forest_model.joblib
│       └── adaboost_model.joblib
│
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── modeling/
│   ├── evaluation/
│   └── utils/
│
├── fastapi_main.py
├── requirements.txt
└── README.md
```

---

## Tech Stack

* **Language:** Python
* **ML & Data:** Scikit-learn, Pandas, NumPy
* **Model Persistence:** joblib
* **Version Control:** Git & GitHub

---

## Key Highlights

* Leakage-safe ML pipeline
* Clean separation of concerns (data, preprocessing, modeling)
* Multiple model experimentation and comparison
* Deployment-ready architecture

---
## How to run 
* using train_models.py file
![Project Demo](model_training.png)

## Next Steps 

* Hyperparameter tuning
* Final model selection
* API hardening and validation
* Frontend / UI integration
* End-to-end deployment
