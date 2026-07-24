import streamlit as st
import joblib
import numpy as np
import os

# --- CRITICAL STREAMLIT FIX: Must be the absolute first Streamlit command ---
st.set_page_config(
    page_title="Liver Disease Prediction",
    page_icon="🩺",
    layout="wide"
)

# ------------------------- Setting Sidebar Color to Blue --------------------------------
st.markdown(
    """
    <style>
    /* Changes sidebar background to Blue (#0D47A1) */
    [data-testid="stSidebar"] {
        background-color: #0D47A1 !important;
    }
    /* Forces all text inside the sidebar to remain crisp white for legibility */
    [data-testid="stSidebar"] __element__, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] j,
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] a,
    [data-testid="stSidebar"] label {
        color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load Models Safely ---
model = joblib.load("liver_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

# --- App Content Structure ---
main_page, main_right = st.columns(2)
with main_page:
    st.title("🩺 Liver Disease Prediction")
    st.write("Input patient physiological data points below to evaluate clinical risk matrix profiles.")
    st.markdown("---")
with main_right:
     st.image("hepatitis.png", width=220)
# Main interface structure splits input columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, step=1)
    gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male (0)" if x == 0 else "Female (1)")
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=22.0)
    alcohol = st.number_input("Alcohol Consumption (Units/Week)", min_value=0.0)
    smoking = st.selectbox("Smoking Status", options=[0, 1], format_func=lambda x: "Non-Smoker (0)" if x == 0 else "Smoker (1)")

with col2:
    genetic_risk = st.number_input("Genetic Risk Factor Score", min_value=0.0)
    physical_activity = st.number_input("Physical Activity (Hours/Week)", min_value=0.0)
    diabetes = st.selectbox("Diabetes History", options=[0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
    hypertension = st.selectbox("Hypertension History", options=[0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
    liver_function_test = st.number_input("Liver Function Test Assay Value", min_value=0.0)

st.markdown("---")

# --- Form Logic & Execution ---
if st.button("Run Diagnostic Prediction", use_container_width=True):
    # Construct feature matrix map
    data = np.array([[
        age,
        gender,
        bmi,
        alcohol,
        smoking,
        genetic_risk,
        physical_activity,
        diabetes,
        hypertension,
        liver_function_test
    ]])

    # Transform through matching pre-fitted pipeline scalar scale
    data_scaled = scaler.transform(data)

    # Compute prediction classification array index
    prediction = model.predict(data_scaled)

    # Output dynamic context results banner
    if prediction[0] == 1:
        st.error("⚠️ **High Risk Configuration Profile:** Liver Disease Detected. Please schedule formal clinical validation paths.")
    else:
        st.success("✅ **Clear Diagnostic Signal:** No Liver Disease Detected within parameters.")

# --- Sidebar Metadata Elements ---
with st.sidebar:
    # Top Logo Integration
    if os.path.exists("hepatitis.png"):
        st.image("hepatitis.png", width=120)
    else:
        st.info("💡 Logo asset not found.")
        
    st.header("🔬 Diagnostic Hub")
    st.write("This secure predictive diagnostic layer uses an offline `scikit-learn` configuration stack.")
    st.markdown("---")
    
    st.subheader("👨‍💻 Application Developer")
    # Profile Picture Integration
    if os.path.exists("IMG-20260704-WA0633.jpg"):
        st.image("IMG-20260704-WA0633.jpg", caption="Sulayman Bah", width=180)
    else:
        st.info("💡 Profile image asset not found.")
        
    st.write("**Engineer:** Sulayman Bah")
    st.write("**Specialization:** ML / DL Applications")
    
    st.markdown("---")
    st.subheader("🔗 Connect With Me")
    st.markdown("[📁 GitHub Profile](https://github.com/bahsulayman689-hash)")
    st.markdown("[💼 LinkedIn Profile](www.linkedin.com/in/sulayman-bah-8a7096423)")
    st.markdown("[📧 Email Support](mailto:bahsulayman689@gmail.com)")
