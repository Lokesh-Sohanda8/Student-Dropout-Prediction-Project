---

## 📌 Student Dropout Prediction System — Project Timeline & Methodology

### 1️⃣ Dataset Collection

The dataset was collected from the **UCI Machine Learning Repository**. The original CSV file was converted into an Excel format to ensure proper column ordering and smooth data import during analysis.

---

# 🎓 Student Dropout Prediction System

## Project Timeline & Methodology

---

## 1️⃣ Data Collection

-   The dataset was collected from the **UCI Machine Learning Repository**, focusing on student academic performance and dropout behavior.
    
-   The original CSV file was converted into an Excel format to ensure smoother data handling and correct import ordering during experimentation.
    

---

## 2️⃣ Environment & Project Setup

-   A dedicated **`requirements.txt`** file was created to manage all necessary libraries and package dependencies.
    
-   All development, experimentation, and analysis were conducted using **Jupyter Notebook**, ensuring transparency, modularity, and reproducibility.
    
-   The project structure was organized to clearly separate data exploration, modeling, tuning, and deployment phases.
    

---

## 3️⃣ Data Loading & Initial Exploration

The dataset was imported and explored using:

-   Structural inspection (`info`)
    
-   Statistical summaries (`describe`)
    
-   Missing value analysis
    
-   Duplicate record checks
    

This step provided a comprehensive understanding of the dataset’s structure, feature distributions, and overall data quality.

---

## 4️⃣ Logical Range & Data Validation Checks

-   Logical range checks were applied to relevant numerical and academic features to ensure internal consistency  
    *(e.g., approved curricular units not exceeding enrolled units)*.
    
-   Additional validation checks and statistical outlier detection were performed.
    

**Outlier Handling Insight:**

> The detected values represented statistically rare observations rather than data errors. Since many flagged features were binary, categorical, or academically meaningful extreme cases (such as high-performing or disengaged students), these values were considered valid and informative. Therefore, no rows were removed, and the dataset was retained in its original form to preserve real-world student behavior.

---

## 5️⃣ Feature Engineering

-   Minimal yet meaningful feature engineering was performed.
    
-   One derived feature was created to enhance academic performance representation.
    
-   The remaining original features were retained, as they already captured sufficient academic, demographic, and socio-economic information for modeling.
    

---

## 6️⃣ Feature Importance & Selection

-   Feature importance was evaluated using a **Random Forest Classifier**.
    
-   Based on importance scores:
    
    -   A **Top 10 feature subset**
        
    -   A **Top 15 feature subset**
        
    
    were extracted from the original ~36 features to:
    
    -   Improve model interpretability
        
    -   Reduce deployment complexity
        
    -   Analyze the impact of feature reduction on predictive performance
        

---

## 7️⃣ Model Development

Multiple classification models were trained and evaluated, including:

-   Logistic Regression
    
-   Decision Tree Classifier
    
-   Random Forest Classifier
    
-   XGBoost
    
-   CatBoost
    
-   Support Vector Machine (SVC)
    
-   Multi-Layer Perceptron (Neural Network)
    

Each model was assessed using consistent evaluation metrics to ensure fair comparison.

---

## 8️⃣ Hyperparameter Tuning

-   Hyperparameter optimization was performed for all models to improve generalization performance.
    
-   **CatBoost** was tuned and retrained separately due to compatibility limitations with `scikit-learn`’s `GridSearchCV`.
    
-   The rationale and implementation details for this separate tuning process were clearly documented in the Jupyter Notebook.
    

---

## 9️⃣ Model Retraining & Final Evaluation

-   All models were retrained using their optimized hyperparameters.
    
-   Final evaluation was conducted using:
    
    -   Accuracy
        
    -   Precision
        
    -   Recall
        
    -   F1-score (with emphasis on the dropout class)
        
-   A comparative analysis was performed to identify the most effective and stable model for student dropout prediction.
    

---

## 🔟 Streamlit Deployment

-   A **Streamlit-based web application** was developed to deploy the final trained model.
    
-   The deployment focused on:
    
    -   Using a **reduced set of 16 high-impact features** to minimize user input complexity
        
    -   Providing an intuitive and user-friendly interface with grouped inputs (academic, demographic, financial)
        
    -   Displaying **dropout probability and risk level (Low / Medium / High)** for decision support
        
-   Sample inputs for low, medium, and high-risk students were documented directly in the deployment file for testing and demonstration purposes.
    

This deployment step ensured the model was not only accurate but also **practically usable in a real-world academic decision-support context**.

---

## 🧠 Outcome

This structured and end-to-end approach ensured:

-   Strong data integrity and validation
    
-   Robust and well-tuned predictive models
    
-   Reduced feature complexity for real-world usability
    
-   Transparent experimentation and reproducibility
    
-   Practical deployability through an interactive web interface
    

The project successfully bridges **machine learning modeling** and **real-world academic decision support**, making it suitable for both technical evaluation and applied use cases.

---