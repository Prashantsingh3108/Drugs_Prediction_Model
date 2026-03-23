import streamlit as st
import numpy as np
import pickle
import os

# -------------------------------
# Page Config (MUST be first)
# -------------------------------
st.set_page_config(page_title="💊 Drug Prediction App", layout="centered")

# -------------------------------
# Load Model (FIXED PATH)
# -------------------------------
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

try:
    model = pickle.load(open(model_path, "rb"))
except:
    st.error("❌ Model file not found. Please check 'model.pkl'")
    st.stop()

# -------------------------------
# UI Design
# -------------------------------
st.title("💊 Drug Classification System")
st.markdown("### Predict the suitable drug based on patient details")
st.write("---")

# -------------------------------
# Input Fields
# -------------------------------
age = st.slider("Age", 15, 80, 30)

sex = st.selectbox("Sex", ["Male", "Female"])
bp = st.selectbox("Blood Pressure", ["LOW", "NORMAL", "HIGH"])
chol = st.selectbox("Cholesterol", ["NORMAL", "HIGH"])
na_to_k = st.slider("Na_to_K Ratio", 5.0, 40.0, 15.0)

# -------------------------------
# Encoding (MUST MATCH TRAINING)
# -------------------------------
sex = 1 if sex == "Male" else 0

bp_map = {"LOW": 0, "NORMAL": 1, "HIGH": 2}
chol_map = {"NORMAL": 0, "HIGH": 1}

bp = bp_map[bp]
chol = chol_map[chol]

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Drug"):

    try:
        input_data = np.array([[age, sex, bp, chol, na_to_k]])
        prediction = model.predict(input_data)[0]

        st.success(f"✅ Predicted Drug: **{prediction}**")

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")

# -------------------------------
# Footer
# -------------------------------
st.write("---")
st.markdown("Made with ❤️ using Streamlit")
