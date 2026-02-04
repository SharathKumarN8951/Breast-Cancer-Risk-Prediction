import streamlit as st
import pandas as pd
import joblib

# ===============================
# Load artifacts
# ===============================
model = joblib.load("mortality_model.pkl")
scaler = joblib.load("scaler.pkl")
important_features = joblib.load("features.pkl")

st.set_page_config(page_title="Breast Cancer Mortality Prediction")
st.title("🩺 Breast Cancer 10-Year Mortality Prediction")

st.write("Predicts **10-year mortality risk** using a trained Decision Tree model.")

# ===============================
# Sidebar Inputs
# ===============================
st.sidebar.header("Patient Inputs")

age = st.sidebar.number_input("Age at Diagnosis", 18, 100, 50)
lumA = st.sidebar.selectbox("Pam50 LumA Subtype", ["No", "Yes"])
cohort = st.sidebar.number_input("Cohort", 0, 10, 1)
npi = st.sidebar.number_input("Nottingham Prognostic Index", 0.0, 10.0, 3.0)
cluster8 = st.sidebar.selectbox("Integrative Cluster 8", ["No", "Yes"])
nodes = st.sidebar.number_input("Positive Lymph Nodes", 0, 50, 0)
tumor_size = st.sidebar.number_input("Tumor Size", 0.0, 200.0, 25.0)
grade = st.sidebar.number_input("Neoplasm Histologic Grade", 1, 3, 2)
mastectomy = st.sidebar.selectbox("Breast Surgery: Mastectomy", ["No", "Yes"])
gene_low = st.sidebar.selectbox("3-Gene Subtype ER+/HER2- Low Prolif", ["No", "Yes"])
mutation = st.sidebar.number_input("Mutation Count", 0, 5000, 50)
ductal = st.sidebar.selectbox("Cancer Type: Invasive Ductal", ["No", "Yes"])

# ===============================
# Build input row
# ===============================
input_data = {
    'Age at Diagnosis': age,
    'Pam50 + Claudin-low subtype_LumA': 1 if lumA == "Yes" else 0,
    'Cohort': cohort,
    'Nottingham prognostic index': npi,
    'Integrative Cluster_8': 1 if cluster8 == "Yes" else 0,
    'Lymph nodes examined positive': nodes,
    'Tumor Size': tumor_size,
    'Neoplasm Histologic Grade': grade,
    'Type of Breast Surgery_Mastectomy': 1 if mastectomy == "Yes" else 0,
    '3-Gene classifier subtype_ER+/HER2- Low Prolif': 1 if gene_low == "Yes" else 0,
    'Mutation Count': mutation,
    'Cancer Type Detailed_Breast Invasive Ductal Carcinoma': 1 if ductal == "Yes" else 0
}

input_df = pd.DataFrame([input_data])
input_df = input_df[important_features]   # enforce correct order

# ===============================
# Scale
# ===============================
input_scaled = scaler.transform(input_df)

# ===============================
# Predict
# ===============================
if st.button("Predict Mortality Risk"):

    prob = model.predict_proba(input_scaled)[0][1]
    pred = int(prob >= 0.5)

    st.subheader("🔍 Prediction Result")
    st.write(f"Predicted Mortality Probability: **{prob:.3f}**")

    if pred == 1:
        st.error("🔴 High Risk of Mortality within 10 Years")
    else:
        st.success("🟢 Low Risk of Mortality within 10 Years")

    st.caption("For academic and educational purposes only.")
