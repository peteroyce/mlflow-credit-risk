"""
FastAPI serving endpoint.

Loads the production model from MLflow and serves predictions.
"""

import logging

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from typing import Literal

from pydantic import BaseModel, Field

from src.config import MLFLOW_TRACKING_URI
from src.data.features import add_derived_features

app = FastAPI(title="Credit Risk API", version="0.1.0")
logger = logging.getLogger(__name__)

model = None
MODEL_NAME = "credit-risk-classifier"


class ApplicantData(BaseModel):
    duration: int = Field(..., ge=1, le=120, description="Loan duration in months")
    credit_amount: float = Field(..., ge=0, description="Loan amount")
    installment_rate: int = Field(..., ge=1, le=4, description="Installment rate as % of income")
    residence_since: int = Field(..., ge=1, le=4, description="Years at current residence")
    age: int = Field(..., ge=18, le=120, description="Applicant age")
    existing_credits: int = Field(..., ge=1, le=4, description="Number of existing credits")
    num_dependents: int = Field(..., ge=1, le=2, description="Number of dependents")
    checking_status: Literal["<0", "0<=X<200", ">=200", "no checking"] = Field(
        ..., description="Status of existing checking account"
    )
    credit_history: Literal[
        "no credits/all paid", "all paid", "existing paid",
        "delayed previously", "critical/other existing credit",
    ] = Field(..., description="Credit history")
    purpose: Literal[
        "new car", "used car", "furniture/equipment", "radio/tv",
        "domestic appliance", "repairs", "education", "vacation",
        "retraining", "business", "other",
    ] = Field(..., description="Purpose of the loan")
    savings_status: Literal[
        "<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings",
    ] = Field(..., description="Savings account/bonds status")
    employment: Literal[
        "unemployed", "<1", "1<=X<4", "4<=X<7", ">=7",
    ] = Field(..., description="Employment duration in years")
    personal_status: Literal[
        "male div/sep", "female div/dep/mar", "male single", "male mar/wid",
    ] = Field(..., description="Personal status and sex")
    other_parties: Literal["none", "co applicant", "guarantor"] = Field(
        ..., description="Other debtors / guarantors"
    )
    property_magnitude: Literal[
        "real estate", "life insurance", "car", "no known property",
    ] = Field(..., description="Property type")
    other_payment_plans: Literal["bank", "stores", "none"] = Field(
        ..., description="Other installment plans"
    )
    housing: Literal["rent", "own", "for free"] = Field(..., description="Housing status")
    job: Literal[
        "unemp/unskilled non res", "unskilled resident",
        "skilled", "high qualif/self emp/mgmt",
    ] = Field(..., description="Job category")
    telephone: Literal["yes", "none"] = Field(..., description="Has telephone")
    foreign_worker: Literal["yes", "no"] = Field(..., description="Is foreign worker")


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
