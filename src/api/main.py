"""
FastAPI serving endpoint.

Loads the production model from MLflow and serves predictions.
"""

import logging

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import MLFLOW_TRACKING_URI
from src.data.features import add_derived_features

app = FastAPI(title="Credit Risk API", version="0.1.0")
logger = logging.getLogger(__name__)

model = None
MODEL_NAME = "credit-risk-classifier"


class ApplicantData(BaseModel):
    duration: int
    credit_amount: float
    installment_rate: int
    residence_since: int
    age: int
    existing_credits: int
    num_dependents: int
    checking_status: str
    credit_history: str
    purpose: str
    savings_status: str
    employment: str
    personal_status: str
    other_parties: str
    property_magnitude: str
    other_payment_plans: str
    housing: str
    job: str
    telephone: str
    foreign_worker: str


class PredictionResponse(BaseModel):
    risk_score: float
    decision: str
    threshold: float


@app.on_event("startup")
def load_model():
    global model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
        logger.info("Production model loaded")
    except Exception as e:
        logger.warning("Could not load production model: %s", e)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: ApplicantData):
    if model is None:
        raise HTTPException(503, "Model not loaded")

    # Build a single-row DataFrame with all input fields
    row = data.model_dump()
    df = pd.DataFrame([row])

    # Apply the same derived-feature engineering used during training
    df = add_derived_features(df)

    # The loaded model is a Pipeline (preprocessor + classifier),
    # so it handles scaling and one-hot encoding internally.
    prob = model.predict_proba(df)[:, 1][0]
    threshold = 0.45

    return PredictionResponse(
        risk_score=round(float(prob), 4),
        decision="approved" if prob < threshold else "denied",
        threshold=threshold,
    )
