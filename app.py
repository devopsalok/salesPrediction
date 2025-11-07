import streamlit as st
import numpy as np
import pickle

# Load model
with open(r'C:\Users\alokk\Documents\GPU_PROJECT\Sales_prediction\salesPrediction\sales_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Sales Predictor", layout="centered")
st.title("📈 Sales Prediction App")

st.markdown("Adjust the sliders to set your ad budget and predict sales.")

# Input sliders
tv = st.slider("TV Budget", 0.0, 300.0, 150.0)
radio = st.slider("Radio Budget", 0.0, 50.0, 25.0)
newspaper = st.slider("Newspaper Budget", 0.0, 100.0, 20.0)

# Predict
if st.button("Predict Sales"):
    input_data = np.array([[tv, radio, newspaper]])
    prediction = model.predict(input_data)
    st.success(f"Estimated Sales: **{prediction[0]:.2f} units**")
