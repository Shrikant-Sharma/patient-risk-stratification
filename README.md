# Patient Risk Stratification Pipeline

End-to-end machine learning pipeline that predicts patient risk of heart disease 
from clinical features. Built to demonstrate the full ML workflow: EDA, feature 
engineering, model selection, explainability, calibration, and deployment.

## Tech Stack
Python · Scikit-learn · XGBoost · PyTorch · SHAP · MLflow · FastAPI · Docker · AWS

## Status
🚧 In development — target completion April 26, 2026

## Project Structure
\`\`\`
patient-risk-stratification/
├── data/              # raw and processed datasets
├── notebooks/         # EDA and modeling notebooks
├── src/               # FastAPI app and reusable modules
├── models/            # trained model artifacts
├── reports/figures/   # SHAP plots, calibration curves
├── requirements.txt   # pinned dependencies
└── README.md
\`\`\`