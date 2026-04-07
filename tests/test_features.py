"""Tests for feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.features import add_derived_features, build_preprocessor


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "duration": [12, 24, 36],
        "credit_amount": [1000, 5000, 10000],
        "installment_rate": [4, 2, 3],
        "residence_since": [1, 4, 2],
        "age": [22, 45, 67],
        "existing_credits": [1, 2, 3],
        "num_dependents": [1, 1, 2],
        "checking_status": ["<0", "0<=X<200", ">=200"],
        "credit_history": ["critical/other-existing-credit", "existing paid", "all paid"],
        "purpose": ["radio/tv", "new car", "furniture/equipment"],
        "savings_status": ["<100", "500<=X<1000", ">=1000"],
        "employment": ["1<=X<4", "4<=X<7", ">=7"],
        "personal_status": ["male single", "female div/dep/mar", "male mar/wid"],
        "other_parties": ["none", "co applicant", "guarantor"],
        "property_magnitude": ["real estate", "life insurance", "car"],
        "other_payment_plans": ["bank", "stores", "none"],
        "housing": ["own", "rent", "for free"],
        "job": ["skilled", "unskilled resident", "high qualif/self emp/mgmt"],
        "telephone": ["none", "yes", "none"],
        "foreign_worker": ["yes", "yes", "no"],
        "class": [0, 1, 0],
    })


def test_derived_features(sample_df):
    result = add_derived_features(sample_df)
    assert "amount_per_month" in result.columns
    assert "age_bin" in result.columns
    assert "credit_load" in result.columns

    # amount_per_month = credit_amount / duration
    expected = 1000 / 12
    assert abs(result["amount_per_month"].iloc[0] - expected) < 0.01


def test_derived_age_bins(sample_df):
    result = add_derived_features(sample_df)
    assert result["age_bin"].iloc[0] == "young"   # 22
    assert result["age_bin"].iloc[1] == "mature"   # 45
    assert result["age_bin"].iloc[2] == "senior"   # 67


def test_preprocessor_output_shape(sample_df):
    preprocessor = build_preprocessor()
    X = sample_df.drop(columns=["class"])
    result = preprocessor.fit_transform(X)
    assert result.shape[0] == 3
    assert result.shape[1] > len(sample_df.columns)  # one-hot expands categoricals


def test_preprocessor_no_nans(sample_df):
    preprocessor = build_preprocessor()
    X = sample_df.drop(columns=["class"])
    result = preprocessor.fit_transform(X)
    assert not np.isnan(result).any()