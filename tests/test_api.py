"""Tests for the API endpoint."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_predict_no_model():
    """Without a loaded model, predict should return 503."""
    sample = {
        "duration": 24,
        "credit_amount": 5000.0,
        "installment_rate": 3,
        "residence_since": 2,
        "age": 35,
        "existing_credits": 1,
        "num_dependents": 1,
        "checking_status": "<0",
        "credit_history": "existing paid",
        "purpose": "new car",
        "savings_status": "<100",
        "employment": "1<=X<4",
        "personal_status": "male single",
        "other_parties": "none",
        "property_magnitude": "real estate",
        "other_payment_plans": "none",
        "housing": "own",
        "job": "skilled",
        "telephone": "yes",
        "foreign_worker": "yes",
    }
    response = client.post("/predict", json=sample)
    assert response.status_code == 503