import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Trained Model
# -----------------------------
model = joblib.load("voting_dropout_model.pkl")

expected_features = model.feature_names_in_

st.set_page_config(page_title="Student Dropout Predictor", layout="centered")

st.title("🎓 Student Dropout Risk Prediction")
st.markdown("### Predict Risk of Student Dropout")

st.divider()

# -----------------------------
# USER INPUTS (Important Ones)
# -----------------------------

age = st.number_input("Age at Enrollment", 15, 60, 20)
admission_grade = st.number_input("Admission Grade", 0.0, 200.0, 120.0)

tuition = st.selectbox("Tuition Fees Up to Date?", ["Yes", "No"])
scholarship = st.selectbox("Scholarship Holder?", ["Yes", "No"])
debtor = st.selectbox("Debtor?", ["Yes", "No"])
gender = st.selectbox("Gender", ["Male", "Female"])

course_group = st.selectbox(
    "Course Group",
    ["STEM", "Health", "Business", "Arts", "Social_Sciences"]
)

st.divider()

# -----------------------------
# Encode Inputs
# -----------------------------

tuition = 1 if tuition == "Yes" else 0
scholarship = 1 if scholarship == "Yes" else 0
debtor = 1 if debtor == "Yes" else 0
gender = 1 if gender == "Male" else 0

# Create empty input dictionary
input_dict = {feature: 0 for feature in expected_features}

# Fill numeric/base features
if "Age at enrollment" in input_dict:
    input_dict["Age at enrollment"] = age

if "Admission grade" in input_dict:
    input_dict["Admission grade"] = admission_grade

if "Tuition fees up to date" in input_dict:
    input_dict["Tuition fees up to date"] = tuition

if "Scholarship holder" in input_dict:
    input_dict["Scholarship holder"] = scholarship

if "Debtor" in input_dict:
    input_dict["Debtor"] = debtor

if "Gender" in input_dict:
    input_dict["Gender"] = gender

# Handle Course_Group dummies
for feature in expected_features:
    if feature.startswith("Course_Group_"):
        if feature == f"Course_Group_{course_group}":
            input_dict[feature] = 1
        else:
            input_dict[feature] = 0

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# -----------------------------
# Prediction
# -----------------------------

if st.button("🔍 Predict Dropout Risk"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()

    if prediction == 1:
        st.error("⚠️ High Risk of Dropout")
    else:
        st.success("✅ Low Risk of Dropout")

    st.markdown(f"### 📊 Probability of Dropout: **{probability:.2%}**")
