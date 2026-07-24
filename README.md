# liver_disease_predictor
# 🩺 Live Predictive Diagnostics: Liver Disease Classifier

An end-to-end Machine Learning web application that predicts liver disease risk profiles based on patient physiological, behavioral, and clinical data metrics. Powered by **Scikit-Learn**, **Joblib**, and **Streamlit**.

![Python Version](https://shields.io)
![Framework](https://shields.io)
![ML Library](https://shields.io)

---

## 🚀 Live Demo
🔗 **Click here to interact with the deployed live application:** *[INSERT YOUR STREAMLIT SHARE LINK HERE]*

---

## 🛠️ System Architecture & Workflow

1. **Data Scaling Processing:** Incoming physiological inputs are passed through a standalone robust pre-fitted serialization scaling pipeline (`scaler.pkl`) to normalize input variance distributions.
2. **Predictive Inference Engine:** The scaled dimensional tensors are ingested into a trained classification model (`liver_disease_model.pkl`) to evaluate the patient’s clinical risk matrix profile.
3. **UI Engine:** Built using a custom-styled responsive Streamlit frontend featuring an integrated high-contrast sidebar design framework.

---

## 📊 Feature Matrix Ingested

The underlying model evaluates the following 10 clinical data points to output its binary risk classification:

* **Age:** Numerical structural baseline validation.
* **Gender:** Binary categorical distribution map.
* **BMI:** Body Mass Index metric boundary validation.
* **Alcohol Consumption:** Scaled continuous parameter assessment (Units/Week).
* **Smoking Status:** Categorical index mapping.
* **Genetic Risk Score:** Predisposition quantitative metrics evaluation.
* **Physical Activity:** Quantifiable healthy metrics threshold tracking (Hours/Week).
* **Diabetes & Hypertension:** Dual-layer diagnostic condition tracking.
* **Liver Function Test Assay:** Critical raw biochemical lab values validation.

---

## 💻 Local Installation & Setup

Follow these installation steps to launch this diagnostic application locally on your native workspace environment:

### 1. Clone the Repository
```bash
git clone https://github.com
cd YOUR_REPOSITORY_NAME
```

### 2. Environment Alignment (Bypassing System Conflicts)
To prevent internal protobuf structure mismatches and `NumPy 2.0` framework collisions, execute the strict environment package adjustments using the command below:
```bash
pip install -r requirements.txt
```

### 3. Required Workspace Inventory Assets
Verify your working directory contains the following file structures before initiating runtime instances:
* `app_app.py` (Core execution source code)
* `liver_disease_model.pkl` (Trained classifier weights matrix)
* `scaler.pkl` (Pre-fitted variance pipelines)
* `logo.png` (Application logo asset)
* `IMG-20260704-WA0633.jpg` (Developer branding asset)

### 4. Boot Up the Dashboard Server
```bash
streamlit run app_app.py
```

---

## 👨‍💻 Developer & Structural Architect
* **Lead AI Engineer:** Sulayman Bah
* **Focus Domain:** Machine Learning / Deep Learning Pipelines & Automated MLOps Dashboards
* **Contact Channel:** [bahsulayman689@gmail.com](mailto:bahsulayman689@gmail.com)

---
*Disclaimer: This system is built as an educational/portfolio machine learning inference platform and does not constitute formal clinical consultation software.*
