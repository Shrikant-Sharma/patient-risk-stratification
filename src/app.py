"""
Patient Risk Stratification API.
Serves a tuned XGBoost model with SHAP explainability.
"""
from mangum import Mangum
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.schemas import (
    PatientFeatures,
    PredictionResponse,
    ExplainResponse,
    FeatureContribution,
    HealthResponse,
)

# Resolve paths relative to this file (works regardless of where uvicorn is launched from)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "xgb_tuned.json"
EXPLAINER_PATH = BASE_DIR / "models" / "shap_explainer.pkl"
CALIBRATION_METADATA_PATH = BASE_DIR / "models" / "calibration_metadata.json"

# Loaded model state — populated at startup
state = {"model": None, "explainer": None,
         "calibration": None, "feature_names": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and metadata at app startup; clean up at shutdown."""
    print("Loading model artifacts...")

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    state["model"] = model

    state["explainer"] = joblib.load(EXPLAINER_PATH)

    with open(CALIBRATION_METADATA_PATH) as f:
        state["calibration"] = json.load(f)

    state["feature_names"] = [
        "age", "trestbps", "chol", "thalach", "oldpeak",
        "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
    ]

    print("Model loaded. API ready.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="Patient Risk Stratification API",
    description="Predicts heart disease risk from clinical features. Tuned XGBoost with SHAP explainability.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _features_to_dataframe(features: PatientFeatures) -> pd.DataFrame:
    """Convert Pydantic model into a single-row DataFrame in the correct column order."""
    row = {name: getattr(features, name) for name in state["feature_names"]}
    return pd.DataFrame([row])


def _risk_tier(probability: float) -> str:
    """Map probability to a clinical risk tier."""
    if probability < 0.30:
        return "LOW"
    if probability < 0.70:
        return "MODERATE"
    return "HIGH"


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health():
    """Verify service is running and model is loaded."""
    return HealthResponse(
        status="ok",
        model_loaded=state["model"] is not None,
        explainer_loaded=state["explainer"] is not None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(features: PatientFeatures):
    """Predict disease probability for a single patient."""
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = _features_to_dataframe(features)
    proba = float(state["model"].predict_proba(X)[0, 1])

    threshold = state["calibration"]["recommended_threshold_clinical"]
    prediction = int(proba >= threshold)

    return PredictionResponse(
        disease_probability=round(proba, 4),
        prediction=prediction,
        risk_tier=_risk_tier(proba),
        threshold_used=round(threshold, 4),
        model_version="xgb_tuned_v1",
    )


@app.post("/explain", response_model=ExplainResponse, tags=["inference"])
async def explain(features: PatientFeatures):
    """Predict + return per-feature SHAP contributions for the prediction."""
    if state["explainer"] is None:
        raise HTTPException(status_code=503, detail="Explainer not loaded")

    X = _features_to_dataframe(features)
    proba = float(state["model"].predict_proba(X)[0, 1])

    shap_explanation = state["explainer"](X)
    shap_values = shap_explanation.values[0]
    base_value = float(shap_explanation.base_values[0]) if hasattr(
        shap_explanation.base_values, '__len__') else float(shap_explanation.base_values)

    contributions = []
    for name, val in zip(state["feature_names"], shap_values):
        contributions.append(FeatureContribution(
            feature=name,
            value=float(getattr(features, name)),
            shap_value=round(float(val), 4),
            pushes_toward="disease" if val > 0 else "healthy",
        ))

    contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)
    top3 = contributions[:3]
    summary = f"Predicted probability of disease: {proba:.2%}. Top contributing features: " + ", ".join(
        f"{c.feature}={c.value:g} (pushes {c.pushes_toward})" for c in top3
    )

    return ExplainResponse(
        disease_probability=round(proba, 4),
        base_value_log_odds=round(base_value, 4),
        contributions=contributions,
        summary=summary,
    )


@app.get("/", tags=["meta"])
async def root():
    """API landing page — points users to /docs."""
    return {
        "service": "Patient Risk Stratification API",
        "docs": "/docs",
        "health": "/health",
    }


# ---------------------------------------------------------------------------
# AWS Lambda handler — translates Lambda events to ASGI for FastAPI.
# Used only when this app is deployed as a Lambda container image.
# Local uvicorn execution ignores this; it just imports cleanly.
# ---------------------------------------------------------------------------

handler = Mangum(app)
