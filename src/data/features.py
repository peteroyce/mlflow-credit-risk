"""
Feature engineering pipeline.

Uses sklearn ColumnTransformer for clean preprocessing of mixed
numeric/categorical features.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    DATA_DIR,
    FEATURE_COLS_CATEGORICAL,
    FEATURE_COLS_NUMERIC,
    RANDOM_SEED,
    TARGET_COL,
)

logger = logging.getLogger(__name__)


def load_data() -> pd.DataFrame:
    path = DATA_DIR / "german_credit.parquet"
    return pd.read_parquet(path)


def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - Scales numeric features (StandardScaler)
    - One-hot encodes categorical features
    """
    numeric_transformer = Pipeline([
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, FEATURE_COLS_NUMERIC),
            ("cat", categorical_transformer, FEATURE_COLS_CATEGORICAL),
        ],
        remainder="drop",
    )
    return preprocessor


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features that might capture risk signals better.
    """
    df = df.copy()

    # Credit amount relative to duration
    if "credit_amount" in df.columns and "duration" in df.columns:
        df["amount_per_month"] = df["credit_amount"] / df["duration"].clip(lower=1)

    # Age-based risk bins
    if "age" in df.columns:
        df["age_bin"] = pd.cut(
            df["age"], bins=[0, 25, 35, 50, 100],
            labels=["young", "mid", "mature", "senior"],
        ).astype(str)

    # Credit utilization proxy
    if "existing_credits" in df.columns and "credit_amount" in df.columns:
        df["credit_load"] = df["existing_credits"] * df["credit_amount"]

    return df


def build_full_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer with column lists that match X
    (including any derived features).
    """
    num_cols = [c for c in FEATURE_COLS_NUMERIC if c in X.columns]
    cat_cols = [c for c in FEATURE_COLS_CATEGORICAL if c in X.columns]

    # Add derived numeric features
    for col in ["amount_per_month", "credit_load"]:
        if col in X.columns:
            num_cols.append(col)

    # Add derived categorical features
    for col in ["age_bin"]:
        if col in X.columns:
            cat_cols.append(col)

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )


def prepare_splits(
    test_size: float = 0.2,
    add_derived: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, ColumnTransformer]:
    """
    Full feature engineering pipeline.

    Returns X_train (DataFrame), X_test (DataFrame), y_train, y_test,
    and an unfitted preprocessor.  The caller is responsible for fitting
    (typically via a Pipeline).
    """
    df = load_data()

    if add_derived:
        df = add_derived_features(df)

    y = df[TARGET_COL].values
    X = df.drop(columns=[TARGET_COL])

    preprocessor = build_full_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED,
    )

    logger.info(
        "Splits: train=%d, test=%d, raw features=%d",
        X_train.shape[0], X_test.shape[0], X_train.shape[1],
    )
    return X_train, X_test, y_train, y_test, preprocessor