"""
Request and response models: the contract between a caller and the service.
The limits and the allowed codes come from the data dictionary, so there is one source for them.
"""
from typing import Dict, List

from pydantic import BaseModel, Field, field_validator

from src.data_dictionary import FIELDS, allowed


def _number(name):
    f = FIELDS[name]
    return Field(..., ge=f["min"], le=f["max"], description=f"{f['meaning']} ({f['unit']})")


def _code(name):
    f = FIELDS[name]
    return Field(..., description=f["meaning"] + ": " + ", ".join(f"{k} = {v}" for k, v in f["values"].items()))


class PatientFeatures(BaseModel):
    """The 13 clinical inputs for one patient. Every field is required."""

    age: float = _number("age")
    sex: int = _code("sex")
    cp: int = _code("cp")
    trestbps: float = _number("trestbps")
    chol: float = _number("chol")
    fbs: int = _code("fbs")
    restecg: int = _code("restecg")
    thalach: float = _number("thalach")
    exang: int = _code("exang")
    oldpeak: float = _number("oldpeak")
    slope: int = _code("slope")
    ca: int = _code("ca")
    thal: int = _code("thal")

    @field_validator("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal")
    @classmethod
    def code_is_known(cls, value, info):
        if value not in allowed(info.field_name):
            raise ValueError(f"{info.field_name} must be one of {allowed(info.field_name)}. See GET /data-dictionary")
        return value

    model_config = {"json_schema_extra": {"example": {
        "age": 60.0, "sex": 1, "cp": 4, "trestbps": 140.0, "chol": 280.0, "fbs": 0, "restecg": 2,
        "thalach": 130.0, "exang": 1, "oldpeak": 2.5, "slope": 2, "ca": 2, "thal": 7}}}


class PredictionResponse(BaseModel):
    disease_probability: float = Field(..., description="The model's probability of disease")
    prediction: int = Field(..., description="1 if the probability is at or above the threshold")
    risk_tier: str = Field(..., description="LOW, MODERATE or HIGH. The cuts are in GET /model")
    threshold_used: float = Field(..., description="The threshold applied")
    model_version: str = Field(..., description="The model that answered")


class FeatureContribution(BaseModel):
    """One input's SHAP contribution to a prediction, in log-odds."""
    feature: str
    value: float
    shap_value: float
    pushes_toward: str  # "disease" or "healthy"


class ExplainResponse(BaseModel):
    disease_probability: float
    base_value_log_odds: float
    contributions: List[FeatureContribution]
    summary: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    explainer_loaded: bool
    model_version: str


class DataDictionaryResponse(BaseModel):
    source: Dict
    target: Dict
    fields: Dict
