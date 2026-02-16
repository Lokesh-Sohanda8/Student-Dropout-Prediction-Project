---
# 🎓 Student Dropout Prediction System

## End-to-End Methodology, Modeling & Deployment Documentation

---

# 1️⃣ Project Objective

The primary objective of this project was to build a robust machine learning system capable of predicting student dropout risk based on academic, demographic, and financial indicators.

The goal was not only high predictive accuracy but also:

* Model stability
* Interpretability
* Reduced deployment complexity
* Real-world usability

---

# 2️⃣ Dataset Collection

The dataset was sourced from the **UCI Machine Learning Repository**, containing student academic records and socio-economic attributes.

The dataset included:

* Academic performance indicators
* Enrollment information
* Financial attributes
* Demographic data
* Course information
* Macro-economic indicators

Initial feature count: ~36–37 features.

---

# 3️⃣ Environment Setup

* Python (Jupyter Notebook)
* scikit-learn
* XGBoost
* CatBoost
* Pandas / NumPy
* Streamlit (Deployment)
* Joblib (Model serialization)

A `requirements.txt` file was maintained for reproducibility.

---

# 4️⃣ Data Exploration & Validation

### Initial Inspection

* `df.info()`
* `df.describe()`
* Missing value check
* Duplicate detection

### Logical Range Checks

We validated:

* Approved units ≤ Enrolled units
* Credited units consistency
* Valid grade ranges
* Binary variable integrity

### Outlier Insight

Statistical outliers were detected, but:

* Most were real academic behaviors
* Many were binary flags
* Extreme academic performance is realistic

Therefore:

> No rows were removed to preserve real-world behavior patterns.

---

# 5️⃣ Feature Engineering & Preprocessing

### A. Target Encoding

The original target variable was converted into:

```
Dropout = 1
Not Dropout = 0
```

---

### B. Handling Course Feature

Original Course was numeric-coded (e.g., 33, 171, 9003).

This was problematic because:

* It introduced artificial ordinal relationships.
* Caused multicollinearity.
* Increased VIF.

Solution:

Courses were grouped into meaningful academic clusters:

* STEM
* Health
* Business
* Arts
* Social Sciences

Then:

```
pd.get_dummies(Course_Group)
```

This improved:

* Interpretability
* Stability
* Reduced artificial collinearity

---

### C. Multicollinearity Handling

We examined:

* Correlation heatmap
* Variance Inflation Factor (VIF)

High multicollinearity observed among:

* Enrolled
* Evaluations
* Approved
* Grade variables

Logical reasoning:

If Enrolled increases → Evaluations increase → Approved increases → Grade changes.

Thus, we removed redundant variables:

* Detailed semester credit columns
* Without evaluations columns

This improved model stability.

---

# 6️⃣ Feature Selection Strategy

Three versions were tested:

* Full feature model (~36 features)
* 25-feature reduced model
* 15-feature optimized model

Feature importance was extracted using:

```
RandomForestClassifier.feature_importances_
```

Goals:

* Reduce deployment complexity
* Improve interpretability
* Maintain performance

Observation:

Performance drop from 36 → 15 features was minimal.

This justified feature reduction.

---

# 7️⃣ Model Development

The following models were trained:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost
* CatBoost
* Support Vector Machine
* MLP (Neural Network)

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1-score (primary metric)
* ROC-AUC

F1-score was prioritized due to class imbalance.

---

# 8️⃣ Hyperparameter Tuning Strategy

### A. Logistic Regression

Tuned:

* C (Regularization strength)
* Penalty (L1 / L2)

Reason:
Controls overfitting and improves generalization.

---

### B. Random Forest

Tuned:

* n_estimators
* max_depth
* min_samples_split
* min_samples_leaf

Reason:
Controls variance and tree complexity.

---

### C. XGBoost

Tuned:

* learning_rate
* max_depth
* n_estimators
* subsample
* colsample_bytree

Reason:
Boosting models are sensitive to learning rate and tree depth.

---

### D. CatBoost

Handled separately due to sklearn compatibility issues.

---

# 9️⃣ Ensemble Strategy — Soft Voting

Final shortlisted models:

* Random Forest
* XGBoost
* Logistic Regression

Soft Voting was implemented:

```
VotingClassifier(voting="soft")
```

### Why Soft Voting?

* Averages probabilities
* Reduces variance
* Improves stability
* Prevents over-reliance on single model

---

# 🔟 Final Model Comparison

| Model  | CV F1  | Test F1 |
| ------ | ------ | ------- |
| RF     | 0.8598 | 0.8937  |
| XGB    | 0.8592 | 0.9054  |
| Voting | 0.8610 | 0.9032  |

Although XGB had slightly higher test F1, Voting had:

* Best CV score
* More stable performance
* Lower variance

Final Model Selected: **Soft Voting Classifier**

---

# 1️⃣1️⃣ Model Saving

Model saved using:

```
joblib.dump(voting_model, "voting_dropout_model.pkl")
```

This ensures portability and deployment compatibility.

---

# 1️⃣2️⃣ Deployment Strategy

Deployment built using **Streamlit**.

Goals:

* Clean UI
* Minimal inputs
* High interpretability
* Practical decision support

---

# 1️⃣3️⃣ Deployment Feature Selection Logic

Although model was trained on full feature space, UI was restricted to:

* Age at Enrollment
* Admission Grade
* Tuition Fees Up to Date
* Scholarship Holder
* Debtor
* Gender
* Course Group

### Why These?

Feature importance analysis showed strongest predictors were:

1. Academic performance
2. Financial stability
3. Debt status
4. Admission grade

These variables capture:

* Academic readiness
* Financial pressure
* Engagement likelihood

Other minor features were auto-filled internally to preserve model compatibility.

---

# 1️⃣4️⃣ Risk Categorization Logic

Probability thresholds:

```
< 30%  → Low Risk
30–60% → Moderate Risk
> 60%  → High Risk
```

This improves interpretability over binary output.

---

# 1️⃣5️⃣ Key Insights

1. Financial instability strongly correlates with dropout.
2. Debtor status significantly increases risk.
3. Admission grade is a strong predictor of persistence.
4. Academic grouping affects dropout likelihood.
5. Feature reduction did not significantly degrade performance.

---

# 1️⃣6️⃣ Challenges Faced

* Multicollinearity issues
* Course numeric encoding problem
* CatBoost sklearn compatibility
* Feature mismatch during deployment
* Maintaining consistency between training and UI inputs

Each challenge was resolved with logical engineering decisions.

---

# 1️⃣7️⃣ Final Outcome

The project successfully delivered:

* Robust predictive model (F1 ≈ 0.90)
* Stable ensemble classifier
* Reduced feature deployment system
* Clean and practical Streamlit interface
* Real-world academic decision support tool

---

# 🏁 Conclusion

This project demonstrates:

* Strong ML engineering practice
* Careful feature selection
* Logical hyperparameter tuning
* Ensemble reasoning
* Practical deployment thinking

It bridges:

**Machine Learning Modeling → Real-World Educational Decision Support**

---
