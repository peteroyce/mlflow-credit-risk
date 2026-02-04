"""
Data ingestion for German Credit dataset.

Downloads from UCI ML Repository, validates schema, and saves
a clean copy locally.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

EXPECTED_SHAPE_MIN = (900, 20)


def download_german_credit() -> pd.DataFrame:
    """
    Fetch German Credit dataset via OpenML.

    Returns a DataFrame with all features and binary target.
    Target: 1 = good credit, 2 = bad credit (we remap to 0/1).
    """
    logger.info("Fetching German Credit dataset from OpenML...")
    data = fetch_openml(name="credit-g", version=1, as_frame=True, parser="auto")
    df = data.frame

    logger.info("Raw shape: %s", df.shape)
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Basic data quality checks."""
    assert df.shape[0] >= EXPECTED_SHAPE_MIN[0], f"Too few rows: {df.shape[0]}"
    assert df.shape[1] >= EXPECTED_SHAPE_MIN[1], f"Too few columns: {df.shape[1]}"

    null_pct = df.isnull().sum() / len(df) * 100
    high_null = null_pct[null_pct > 50]
    if len(high_null) > 0:
        logger.warning("Columns with >50%% nulls: %s", high_null.to_dict())

    # Remap target: good=0, bad=1 (we're predicting risk of default)
    if "class" in df.columns:
        df["class"] = (df["class"] == "bad").astype(int)
        logger.info(
            "Target distribution: good=%d (%.1f%%), bad=%d (%.1f%%)",
            (df["class"] == 0).sum(),
            (df["class"] == 0).mean() * 100,
            (df["class"] == 1).sum(),
            (df["class"] == 1).mean() * 100,
        )

    return df


def run():
    """Download, validate, and save the dataset."""
    logging.basicConfig(level=logging.INFO)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = download_german_credit()
    df = validate(df)

    out_path = DATA_DIR / "german_credit.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Saved to %s (%d rows, %d cols)", out_path, *df.shape)


if __name__ == "__main__":
    run()


DEFAULT_5 = 35
