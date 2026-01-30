import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download

# ==========================================
# 1. Page Configuration & Model Loading
# ==========================================
st.set_page_config(page_title="Engine Predictive Maintenance", layout="centered")

REPO_ID = "dpanchali/predictive_maintenance_model"
FILENAME = "predictive_maintenance_model.joblib"

@st.cache_resource
def load_model():
    """Download and load the model from Hugging Face Hub."""
    try:
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ==========================================
# 2. UI Layout
# ==========================================
st.title("🚢 Engine Condition Predictor")
st.markdown("""
This application uses a trained **XGBoost** model to predict engine health
based on sensor patterns.
""")

st.header("Input Engine Sensor Data")

col1, col2 = st.columns(2)

with col1:
    engine_rpm = st.number_input("Engine RPM", min_value=0, max_value=10000, value=700)
    lub_oil_pressure = st.number_input("Lub Oil Pressure (bar)", min_value=0.0, value=2.5, format="%.4f")
    fuel_pressure = st.number_input("Fuel Pressure (bar)", min_value=0.0, value=11.8, format="%.4f")

with col2:
    coolant_pressure = st.number_input("Coolant Pressure (bar)", min_value=0.0, value=3.2, format="%.4f")
    lub_oil_temp = st.number_input("Lub Oil Temp (°C)", min_value=0.0, value=84.1, format="%.4f")
    coolant_temp = st.number_input("Coolant Temp (°C)", min_value=0.0, value=81.6, format="%.4f")

# ==========================================
# 3. Prediction Logic
# ==========================================
if st.button("Predict Engine Condition", type="primary"):
    if model is not None:
        # Calculate engineered features right before prediction
        load_index = float(engine_rpm * fuel_pressure / 100)
        thermal_stress = float(coolant_temp - lub_oil_temp)

        # Create DataFrame with EXACT column names and order
        input_data = pd.DataFrame([[
            engine_rpm,
            lub_oil_pressure,
            fuel_pressure,
            coolant_pressure,
            lub_oil_temp,
            coolant_temp,
            load_index,
            thermal_stress
        ]], columns=[
            'Engine rpm', 'Lub oil pressure', 'Fuel pressure', 'Coolant pressure',
            'lub oil temp', 'Coolant temp', 'load_index', 'thermal_stress'
        ])

        # Perform prediction (Binary Result Only)
        prediction = model.predict(input_data)[0]

        st.divider()
        st.subheader("Results")

        # Visual Feedback based on Prediction
        if prediction == 0: # Assuming 0 is Good
             st.success("**Status: Engine is in Good Condition**")
        else:
             st.error("**Status: Maintenance Required (Potential Fault)**")

    else:
        st.error("Model could not be loaded. Check your Hugging Face Repo ID.")

# ==========================================
# 4. Sidebar Information
# ==========================================
with st.sidebar:
    st.markdown("### Model Details")
    st.text(f"Repo: {REPO_ID}")
    st.markdown("---")
    st.write("Calculated Features:")
    st.caption(f"Load Index: {engine_rpm * fuel_pressure / 100:.2f}")
    st.caption(f"Thermal Stress: {coolant_temp - lub_oil_temp:.2f}")
