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

import os

data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'results.csv')

if os.path.exists(data_path):
    st.success("Loaded evaluation results.")
    df = pd.read_csv(data_path)
    
    if attack_choice == "None (Clean)":
        st.write(f"#### Baseline Clean Accuracy for {model_choice}")
        clean_data = df[(df['Model'] == model_choice) & (df['Epsilon'] == 0.0)].drop_duplicates(subset=['Precision'])
        
        # Display clean accuracy
        st.bar_chart(data=clean_data.set_index('Precision')['Clean Acc (%)'], height=300)
        st.dataframe(clean_data[['Precision', 'Clean Acc (%)']])
    else:
        # Filter data for specific attack
        filtered = df[(df['Model'] == model_choice) & (df['Attack'] == attack_choice)]
        
        st.write(f"#### {model_choice} under {attack_choice} Attack")
        
        # Get precision specific data
        fp32_data = filtered[filtered['Precision'] == 'FP32']
        fp16_data = filtered[filtered['Precision'] == 'FP16']
        int8_data = filtered[filtered['Precision'] == 'INT8']
        
        fig, ax = plt.subplots(figsize=(8, 3))
        if not fp32_data.empty:
            ax.plot(fp32_data['Epsilon'], fp32_data['ASR (%)'], marker='o', label='FP32')
        if not fp16_data.empty:
            ax.plot(fp16_data['Epsilon'], fp16_data['ASR (%)'], marker='s', label='FP16')
        if not int8_data.empty:
            ax.plot(int8_data['Epsilon'], int8_data['ASR (%)'], marker='x', label='INT8')
            
        ax.set_xlabel("Epsilon")
        ax.set_ylabel("Attack Success Rate (%)")
        ax.legend()
        st.pyplot(fig)
        
        # Show data table
        st.write("#### Raw Attack Metrics")
        st.dataframe(filtered[['Precision', 'Epsilon', 'ASR (%)']].sort_values(by=['Precision', 'Epsilon']))
else:
    st.info("Run the evaluation scripts to generate data, which will populate these charts.")
