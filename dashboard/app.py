import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="VisionGuard Dashboard", layout="wide")

st.title("VisionGuard: Model Quantisation & Adversarial Resilience")
st.markdown("Identifying the exact tipping point where compressed computer vision models become vulnerable to gradient attacks.")

st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox("Select Model", ["ViT-B/16", "ConvNeXt-Tiny"])
precision_choice = st.sidebar.selectbox("Select Precision", ["FP32", "FP16", "INT8"])
attack_choice = st.sidebar.selectbox("Select Attack", ["None (Clean)", "FGSM", "PGD"])
epsilon = st.sidebar.slider("Perturbation Budget (Epsilon)", 0.0, 16.0, 8.0, 1.0) / 255.0

st.write("### Attack Success Rate (ASR) vs. Perturbation Budget")

# Placeholder for actual data visualization
st.info("Run the evaluation scripts to generate data, which will populate these charts.")

# Example dummy chart
st.write("#### Example Metric Comparison")
dummy_data = pd.DataFrame({
    'Epsilon': [0, 2, 4, 8, 16],
    'FP32 ASR (%)': [0, 15, 35, 75, 95],
    'INT8 ASR (%)': [0, 25, 50, 85, 98]
})
fig, ax = plt.subplots()
ax.plot(dummy_data['Epsilon'], dummy_data['FP32 ASR (%)'], marker='o', label='FP32')
ax.plot(dummy_data['Epsilon'], dummy_data['INT8 ASR (%)'], marker='x', label='INT8')
ax.set_xlabel("Epsilon")
ax.set_ylabel("Attack Success Rate (%)")
ax.legend()
st.pyplot(fig)
