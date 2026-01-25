import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from pathlib import Path

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Auto MPG Predictor",
    page_icon="🚗",
    layout="centered"
)

# 2. MODEL PATH 
MODEL_PATH = Path(__file__).parent / "mpg_predictor.cbm"

# 3. LOAD MODEL 
@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# 4. SIDEBAR INPUTS
st.sidebar.header("🔧 Vehicle Configuration")
st.sidebar.markdown("Adjust the sliders below to match your car specs.")

def user_input_features():
    weight = st.sidebar.slider("Weight (lbs)", 1500, 5200, 3000)
    horsepower = st.sidebar.slider("Horsepower", 40, 250, 130)
    displacement = st.sidebar.slider("Displacement (cu. in.)", 60, 460, 200)
    acceleration = st.sidebar.slider("Acceleration (0–60 time)", 8.0, 25.0, 15.0)
    model_year = st.sidebar.slider("Model Year (70 = 1970)", 70, 82, 79)
    cylinders = st.sidebar.selectbox("Cylinders", [3, 4, 5, 6, 8], index=1)

    origin_display = ["USA", "Europe", "Asia"]
    origin_map = {"USA": "1", "Europe": "2", "Asia": "3"}
    origin = origin_map[st.sidebar.selectbox("Origin", origin_display)]

    return pd.DataFrame({
        'cylinders': [cylinders],
        'displacement': [displacement],
        'horsepower': [horsepower],
        'weight': [weight],
        'acceleration': [acceleration],
        'model_year': [model_year],
        'origin': [origin]
    })

input_df = user_input_features()

# 5. MAIN UI
st.title("🚗 Auto MPG Predictor")
st.markdown("Predict Miles Per Gallon (MPG) using vehicle specifications.")
st.divider()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Core Specs")
    st.write(input_df[['weight', 'horsepower', 'displacement']].T)

with col2:
    st.subheader("Additional Specs")
    st.write(input_df[['acceleration', 'model_year', 'cylinders', 'origin']].T)

# 6. PREDICTION
if st.button("🚀 Calculate MPG", type="primary"):

    df = input_df.copy()

    # Feature engineering
    df['cylinders'] = df['cylinders'].astype(float)
    df['power_to_weight'] = df['horsepower'] / df['weight']
    df['disp_per_cyl'] = df['displacement'] / df['cylinders']
    df['acc_per_hp'] = df['acceleration'] / df['horsepower']

    # Convert categoricals
    df['cylinders'] = df['cylinders'].astype(int).astype(str)
    df['origin'] = df['origin'].astype(str)

    
    FEATURE_ORDER = [
        'cylinders',
        'displacement',
        'horsepower',
        'weight',
        'acceleration',
        'model_year',
        'origin',
        'power_to_weight',
        'disp_per_cyl',
        'acc_per_hp'
    ]

    df = df[FEATURE_ORDER]

    # Predict
    prediction = model.predict(df)[0]
    prediction = np.clip(prediction, 5, 60)

    # Display
    st.divider()
    st.subheader("Prediction Result")
    st.metric("Predicted Efficiency", f"{prediction:.2f} MPG")
    st.caption("Reference scale: 0–50 MPG")
    st.progress(min(prediction / 50, 1.0))
