"""
Patient Risk Stratification API.

Serves the XGBoost model with a SHAP explanation for each prediction, a data dictionary, a
request log and a drift check. Everything it needs is in the repository: the trees
(models/xgb_v2.json) and one metadata file. It needs xgboost, numpy, FastAPI and pydantic.

A portfolio project on a public research file of 303 patients. Not a medical device, and not
for use on real patients.
"""
import json
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import xgboost as xgb
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src import data_dictionary, monitoring
from src.schemas import (DataDictionaryResponse, ExplainResponse, FeatureContribution, HealthResponse,
                         PatientFeatures, PredictionResponse)

BASE_DIR = Path(__file__).resolve().parent.parent
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

state = {"booster": None, "meta": None}


def load():
    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    booster = xgb.Booster()
    booster.load_model(str(BASE_DIR / "models" / meta["model_file"]))
    state["booster"], state["meta"] = booster, meta


@asynccontextmanager
async def lifespan(app: FastAPI):
    load()
    yield


app = FastAPI(
    title="Patient Risk Stratification API",
    description="Heart disease risk from 13 clinical inputs: XGBoost with a SHAP explanation, a data dictionary and a drift check. "
                "A portfolio project on a public research file. Not for use on real patients.",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _ready():
    if state["booster"] is None:          # Lambda can call the handler without the startup event
        try:
            load()
        except Exception as error:
            raise HTTPException(status_code=503, detail=f"Model not loaded: {error}")
    return state["booster"], state["meta"]


def _matrix(features, meta):
    """One row, in the order the model was trained in."""
    row = [float(getattr(features, name)) for name in meta["features"]]
    return xgb.DMatrix(np.array([row], dtype=np.float32), feature_names=meta["features"])


def calibrate(p, cal):
    """The calibration stored in the metadata, as arithmetic."""
    if cal["method"] == "platt":
        p = min(max(p, 1e-6), 1 - 1e-6)
        return float(1 / (1 + np.exp(-(cal["a"] * np.log(p / (1 - p)) + cal["b"]))))
    if cal["method"] == "isotonic":
        return float(np.interp(p, cal["x"], cal["y"]))
    return float(p)


def _tier(probability, meta):
    if probability < meta["threshold"]:
        return "LOW"
    return "MODERATE" if probability < meta["tiers"]["upper_cut"] else "HIGH"


def _score(features, endpoint, check=None):
    booster, meta = _ready()
    raw = float(booster.predict(_matrix(features, meta))[0])
    probability = calibrate(raw, meta["calibration"])
    flagged = int(probability >= meta["threshold"])
    if check is None:                     # the weekly check marks its own requests, so they stay out of the drift window
        monitoring.log_request({n: float(getattr(features, n)) for n in meta["features"]}, raw, probability, flagged, meta["model_version"], endpoint)
    return probability, flagged


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health():
    """The service is up and the model is loaded."""
    booster, meta = _ready()
    return HealthResponse(status="ok", model_loaded=True, explainer_loaded=True, model_version=meta["model_version"])


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(features: PatientFeatures, x_check: Optional[str] = Header(default=None)):
    """The probability of disease for one patient, and whether it crosses the threshold."""
    probability, flagged = _score(features, "predict", x_check)
    meta = state["meta"]
    return PredictionResponse(disease_probability=round(probability, 4), prediction=flagged, risk_tier=_tier(probability, meta),
                              threshold_used=round(meta["threshold"], 4), model_version=meta["model_version"])


@app.post("/explain", response_model=ExplainResponse, tags=["inference"])
async def explain(features: PatientFeatures, x_check: Optional[str] = Header(default=None)):
    """The prediction with each input's SHAP contribution. XGBoost computes the exact tree SHAP values itself."""
    probability, _ = _score(features, "explain", x_check)
    booster, meta = state["booster"], state["meta"]
    contribs = booster.predict(_matrix(features, meta), pred_contribs=True)[0]      # 13 inputs, then the base value
    items = [FeatureContribution(feature=name, value=float(getattr(features, name)), shap_value=round(float(v), 4),
                                 pushes_toward="disease" if v > 0 else "healthy")
             for name, v in zip(meta["features"], contribs[:-1])]
    items.sort(key=lambda c: abs(c.shap_value), reverse=True)
    summary = f"Predicted probability of disease: {probability:.2%}. Top contributing features: " + ", ".join(
        f"{c.feature}={c.value:g} (pushes {c.pushes_toward})" for c in items[:3])
    return ExplainResponse(disease_probability=round(probability, 4), base_value_log_odds=round(float(contribs[-1]), 4),
                           contributions=items, summary=summary)


@app.get("/data-dictionary", response_model=DataDictionaryResponse, tags=["meta"])
async def dictionary():
    """What each input means, its unit, the values it may take, and the range seen in training."""
    _, meta = _ready()
    fields = {}
    for name in meta["features"]:
        fields[name] = dict(data_dictionary.FIELDS[name])
        if name in meta["training_range"]:
            fields[name]["seen_in_training"] = meta["training_range"][name]
    return DataDictionaryResponse(source=data_dictionary.SOURCE, target=data_dictionary.TARGET, fields=fields)


@app.get("/model", tags=["meta"])
async def model_facts():
    """The served model's facts: recipe, threshold, tiers, measured scores, versions."""
    _, meta = _ready()
    return {k: v for k, v in meta.items() if k != "drift_reference"}


@app.get("/monitoring/drift", tags=["monitoring"])
async def drift():
    """Compare the latest requests in this instance's log with the training patients."""
    _, meta = _ready()
    return monitoring.drift_report(monitoring.read_log(), meta["drift_reference"])


@app.get("/", tags=["meta"])
async def root():
    return {"service": "Patient Risk Stratification API", "docs": "/docs", "health": "/health",
            "data_dictionary": "/data-dictionary", "model": "/model", "drift": "/monitoring/drift",
            "note": "A portfolio project on a public research file. Not for use on real patients."}


# AWS Lambda: Mangum turns a Lambda event into a request for this app. It is only installed in the Lambda image.
try:
    from mangum import Mangum
    handler = Mangum(app)
except ImportError:
    handler = None
