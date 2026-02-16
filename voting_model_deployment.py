import streamlit as st
import pandas as pd
import joblib

# ==========================================================
# Page Configuration (MUST be first Streamlit command)
# ==========================================================

st.set_page_config(
    page_title="Student Dropout Risk Predictor",
    page_icon="🎓",
    layout="centered"
)

# ==========================================================
# Load Model (Cached)
# ==========================================================

@st.cache_resource
def load_model():
    return joblib.load("xgb_model_dropout.pkl")

model = load_model()
expected_features = model.feature_names_in_

# ==========================================================
# UI HEADER
# ==========================================================

st.title("🎓 Student Dropout Risk Prediction")
st.markdown("### Predict the likelihood of student dropout")
st.write("Provide key academic and financial details below.")

st.divider()

# ==========================================================
# USER INPUT SECTION
# ==========================================================

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age at Enrollment", 15, 60, 20)
    admission_grade = st.number_input("Admission Grade", 0.0, 200.0, 120.0)

with col2:
    tuition = st.selectbox("Tuition Fees Up to Date?", ["Yes", "No"])
    scholarship = st.selectbox("Scholarship Holder?", ["Yes", "No"])
    debtor = st.selectbox("Debtor?", ["Yes", "No"])
    gender = st.selectbox("Gender", ["Male", "Female"])

course_group = st.selectbox(
    "Course Group",
    ["STEM", "Health", "Business", "Arts", "Social_Sciences"]
)

st.divider()

# ==========================================================
# ENCODING
# ==========================================================

tuition = 1 if tuition == "Yes" else 0
scholarship = 1 if scholarship == "Yes" else 0
debtor = 1 if debtor == "Yes" else 0
gender = 1 if gender == "Male" else 0

# Create full feature dictionary
input_dict = {feature: 0 for feature in expected_features}

# Fill numeric/base features
feature_mapping = {
    "Age at enrollment": age,
    "Admission grade": admission_grade,
    "Tuition fees up to date": tuition,
    "Scholarship holder": scholarship,
    "Debtor": debtor,
    "Gender": gender
}

for feature, value in feature_mapping.items():
    if feature in input_dict:
        input_dict[feature] = value

# Handle Course Group One-Hot Encoding
for feature in expected_features:
    if feature.startswith("Course_Group_"):
        input_dict[feature] = 1 if feature == f"Course_Group_{course_group}" else 0

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# ==========================================================
# PREDICTION SECTION
# ==========================================================

if st.button("🔍 Predict Dropout Risk"):

    probability = model.predict_proba(input_df)[0][1]

    # Custom Risk Segmentation
    if probability >= 0.65:
        risk_label = "🔴 High Risk"
        st.error("⚠️ High Risk of Dropout")
    elif probability >= 0.35:
        risk_label = "🟡 Medium Risk"
        st.warning("⚠️ Moderate Risk of Dropout")
    else:
        risk_label = "🟢 Low Risk"
        st.success("✅ Low Risk of Dropout")

    st.divider()

    st.markdown(f"## {risk_label}")
    st.markdown(f"### 📊 Probability of Dropout: **{probability:.2%}**")

    # Risk Interpretation
    if probability >= 0.65:
        st.write("This student shows strong indicators associated with dropout risk. Immediate intervention recommended.")
    elif probability >= 0.35:
        st.write("This student may require monitoring and academic/financial support.")
    else:
        st.write("This student shows strong persistence indicators.")

