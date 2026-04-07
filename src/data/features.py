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


def prepare_splits(
    test_size: float = 0.2,
    add_derived: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer]:
    """
    Full feature engineering pipeline.

    Returns X_train, X_test, y_train, y_test, fitted preprocessor.
    """
    df = load_data()

    if add_derived:
        df = add_derived_features(df)

    y = df[TARGET_COL].values
    X = df.drop(columns=[TARGET_COL])

    # Update feature lists if derived features exist
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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED,
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    logger.info(
        "Splits: train=%d (%d features), test=%d",
        X_train_proc.shape[0], X_train_proc.shape[1], X_test_proc.shape[0],
    )
    return X_train_proc, X_test_proc, y_train, y_test, preprocessor