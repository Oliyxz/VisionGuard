# VisionGuard: Model Quantisation & Adversarial Resilience

## Overview
This project investigates the intersection of machine learning and cyber security by evaluating how post-training quantisation affects the structural resilience of Vision Transformers (ViT) and Convolutional Neural Networks (ConvNeXt).

## Structure
- `src/`: Core Python modules for data loading, model setup, quantisation, and attacks.
- `dashboard/`: Streamlit interactive dashboard.
- `notebooks/`: Jupyter notebooks for exploratory analysis.
- `data/`: Datasets (CIFAR-100, Tiny-ImageNet).
- `models/`: Downloaded and quantised models.

## Setup
1. Create a Python virtual environment: `python -m venv venv`
2. Activate the virtual environment.
3. Install dependencies: `pip install -r requirements.txt`

## Phases
1. Environment setup and data loading.
2. Baseline accuracy evaluation.
3. Post-training static quantisation to INT8.
4. Adversarial attack generation (FGSM, PGD).
5. Data visualisation dashboard.
