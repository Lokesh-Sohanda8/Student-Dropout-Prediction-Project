# 🎓 Student Dropout Prediction System

### End-to-End Machine Learning & Deployment Project

---

## 📌 Project Overview

This project builds a complete **Machine Learning system** to predict student dropout risk using academic, demographic, and financial indicators.

The goal was not only high predictive performance, but also:

* ✔ Model stability
* ✔ Reduced feature complexity
* ✔ Deployment-ready architecture
* ✔ Practical academic decision support

The final system delivers:

* Dropout classification (High / Moderate / Low Risk)
* Probability of dropout
* Interactive Streamlit web interface

---

## 📊 Dataset

* Source: **UCI Machine Learning Repository**
* Domain: Higher Education Student Performance
* Initial Features: ~36–37 variables
* Final Optimized Features: 25 → 15 → Deployment subset

Dataset includes:

* Academic performance indicators
* Enrollment behavior
* Financial status
* Demographics
* Course information
* Economic indicators

---

## ⚙️ Project Workflow

### 1️⃣ Data Exploration & Validation

* Structural inspection (`info`, `describe`)
* Missing value check
* Duplicate detection
* Logical consistency validation
* Statistical outlier analysis

📌 Important Insight:
Outliers were retained since they represented real-world academic behavior rather than data errors.

---

### 2️⃣ Feature Engineering

* Binary encoding of target variable
* Course grouping into meaningful clusters:

  * STEM
  * Health
  * Business
  * Arts
  * Social Sciences
* One-hot encoding of grouped course categories
* Removal of redundant semester-level academic features

📌 Reason:
Reduce multicollinearity and improve interpretability.

---

### 3️⃣ Feature Selection Strategy

Three model versions were developed:

* Full Feature Model (~36 features)
* Reduced 25 Feature Model
* Optimized 15 Feature Model

Feature importance extracted using Random Forest.

Observation:

> Performance degradation from 36 → 15 features was minimal.

This justified feature reduction for deployment simplicity.

---

## 🤖 Models Trained

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost
* CatBoost
* Support Vector Machine (SVC)
* Multi-Layer Perceptron

Evaluation Metrics:

* Accuracy
* Precision
* Recall
* **F1-score (Primary Metric)**
* ROC-AUC

F1-score was prioritized due to class imbalance.

---

## 🔧 Hyperparameter Tuning

Each model was tuned using GridSearchCV or RandomizedSearchCV.

Examples:

**Random Forest**

* n_estimators
* max_depth
* min_samples_split
* min_samples_leaf

**XGBoost**

* learning_rate
* max_depth
* n_estimators
* subsample
* colsample_bytree

**Logistic Regression**

* Regularization (C)
* Penalty type

CatBoost was tuned separately due to sklearn compatibility.

---

## 🏆 Final Model Selection

Final shortlisted models:

| Model           | CV F1      | Test F1    |
| --------------- | ---------- | ---------- |
| Random Forest   | 0.8598     | 0.8937     |
| XGBoost         | 0.8592     | 0.9054     |
| **Soft Voting** | **0.8610** | **0.9032** |

Although XGBoost achieved slightly higher test F1,
Soft Voting was selected due to:

* Best cross-validation performance
* Greater stability
* Reduced variance
* Better generalization consistency

Final Model: **Soft Voting Classifier**

---

## 📈 Evaluation Curves

The final model was evaluated using:

* ROC Curve (AUC ≈ 0.96+)
* Precision–Recall Curve (AP ≈ 0.88–0.91)

Precision-Recall curve was emphasized due to class imbalance.

---

## 💾 Model Serialization

The final trained model was saved using:

```python
joblib.dump(voting_model, "voting_dropout_model.pkl", compress=('xz', 3))
```

Compression applied to reduce model size.

---

# 🚀 Deployment (Streamlit Web App)

An interactive web application was developed using **Streamlit**.

---

## 🎯 Deployment Feature Selection

Although the model was trained on the full optimized feature set,
the UI was intentionally simplified to require only high-impact inputs:

* Age at Enrollment
* Admission Grade
* Tuition Fees Up to Date
* Scholarship Holder
* Debtor
* Gender
* Course Group

### Why These?

Feature importance analysis revealed strongest predictors were:

1. Academic readiness
2. Financial stability
3. Debt status
4. Course category

This ensured:

* Minimal user input
* Maximum predictive power
* Practical usability

Remaining required model features are auto-filled internally to preserve compatibility.

---

## 📊 Risk Categorization

Probability thresholds:

* **< 30% → Low Risk**
* **30–60% → Moderate Risk**
* **> 60% → High Risk**

This improves interpretability beyond binary classification.

---

## 🧠 Key Insights

* Financial instability strongly correlates with dropout.
* Debtor status significantly increases dropout probability.
* Admission grade is a strong predictor of persistence.
* Academic grouping influences completion likelihood.
* Feature reduction improves deployability without harming performance.

---

## 🛠 Technologies Used

* Python
* scikit-learn
* XGBoost
* CatBoost
* Pandas / NumPy
* Matplotlib / Seaborn
* Streamlit
* Joblib

---

## 🏁 Final Outcome

This project successfully bridges:

**Machine Learning Engineering → Real-World Academic Decision Support**

It demonstrates:

* Strong model experimentation
* Logical feature engineering
* Ensemble modeling
* Hyperparameter tuning
* Deployment-focused thinking
* Production-ready ML workflow

---

## 📌 Future Improvements

* SHAP-based explainability
* Automated threshold tuning
* Model monitoring system
* Cloud deployment (Streamlit Cloud / AWS / Azure)
* REST API integration

---

## 👤 Author

- Name: Lokesh Sohanda
- Project Name: Student Dropout Prediction System
- Type: Machine Learning Project
- End-to-End Model + Deployment Implementation

---
