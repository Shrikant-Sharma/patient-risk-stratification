"""
Pydantic schemas for request and response models.
Defines the contract between API consumers and the service.
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class PatientFeatures(BaseModel):
    """Clinical features for a single patient prediction request."""

    age: float = Field(..., ge=18, le=100, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=50, le=250,
                            description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=100, le=700,
                        description="Serum cholesterol (mg/dL)")
    fbs: int = Field(..., ge=0, le=1,
                     description="Fasting blood sugar > 120 mg/dL (1 = yes)")
    restecg: int = Field(..., ge=0, le=2,
                         description="Resting ECG results (0, 1, 2)")
    thalach: float = Field(..., ge=50, le=250,
                           description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1,
                       description="Exercise-induced angina (1 = yes)")
    oldpeak: float = Field(..., ge=0, le=10,
                           description="ST depression induced by exercise")
    slope: int = Field(..., ge=1, le=3,
                       description="Slope of peak exercise ST segment")
    ca: int = Field(..., ge=0, le=3,
                    description="Number of major vessels (0-3)")
    thal: int = Field(..., description="Thalassemia type (3 = normal, 6 = fixed defect, 7 = reversible)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 60.0,
                "sex": 1,
                "cp": 4,
                "trestbps": 140.0,
                "chol": 280.0,
                "fbs": 0,
                "restecg": 2,
                "thalach": 130.0,
                "exang": 1,
                "oldpeak": 2.5,
                "slope": 2,
                "ca": 2,
                "thal": 7
            }
        }
    }


class PredictionResponse(BaseModel):
    """Response from the /predict endpoint."""

    disease_probability: float = Field(...,
                                       description="P(disease) from the model")
    prediction: int = Field(...,
                            description="Binary prediction at clinical threshold")
    risk_tier: str = Field(..., description="LOW, MODERATE, or HIGH")
    threshold_used: float = Field(...,
                                  description="Clinical threshold applied")
    model_version: str = Field(...,
                               description="Identifier for the deployed model")


class FeatureContribution(BaseModel):
    """One feature's SHAP contribution to a prediction."""
    feature: str
    value: float
    shap_value: float
    pushes_toward: str  # "disease" or "healthy"


class ExplainResponse(BaseModel):
    """Response from the /explain endpoint."""
    disease_probability: float
    base_value_log_odds: float
    contributions: List[FeatureContribution]
    summary: str  # human-readable interpretation


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""
    status: str
    model_loaded: bool
    explainer_loaded: bool
