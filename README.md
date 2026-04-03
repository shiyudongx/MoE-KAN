# Interpretable Mixture-of-Experts Kolmogorov-Arnold Network for Predicting Light Yield and Discovering Novel Inorganic Scintillators 🚀

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A high-performance PyTorch implementation of **Mixture-of-Experts (MoE)** combined with **Kolmogorov-Arnold Networks (KAN)**. This project is specifically designed for complex regression tasks and feature importance analysis in **Materials Science** and **Chemical Informatics**.

---

## 🌟 Project Highlights

- **Advanced Architecture**: Combines the interpretability of KAN with the representational power of MoE (Gating Network + Multiple KAN Experts).
- **Explainable AI (XAI)**: Integrated SHAP analysis to reveal the non-linear contribution of each feature.
- **Symbolic Regression**: Built-in support for genetic programming-based symbolic regression to derive analytical formulas.

---

## 📊 Framework Architecture

Below is the conceptual workflow of the MoE-KAN model, illustrating how the gating network routes input features to specialized KAN experts.



<p align="center">
  <img src="workflow.png" alt="MoE-KAN Architecture" width="800">
</p>

---

##  Repository Structure

```text
MoE-KAN/
├── data/               # Datasets (.csv) and prediction results
├── models/             # Saved model weights (.pth) and scalers (.pkl)
├── src/                # Core implementation of KAN and MoE layers
├── notebooks/          # Exploratory Data Analysis and Jupyter experiments
├── train.py            # Main training script
├── evaluate.py         # Model performance evaluation script
├── requirements.txt    # Environment dependencies
└── README.md           # Project documentation

---
# Clone the repository
git clone [https://github.com/shiyudongx/MoE-KAN.git](https://github.com/shiyudongx/MoE-KAN.git)

# Enter the project directory
cd MoE-KAN

# Install dependencies
pip install -r requirements.txt

Run the training script to normalize data and train the MoE-KAN network:
python train.py

Evaluate the model's performance on the test set
python evaluate.py


