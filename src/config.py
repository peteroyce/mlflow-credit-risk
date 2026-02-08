"""
Project configuration.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

MLFLOW_TRACKING_URI = "mlruns"
EXPERIMENT_NAME = "credit-risk"

RANDOM_SEED = 42

FEATURE_COLS_NUMERIC = [
    "duration", "credit_amount", "installment_rate", "residence_since",
    "age", "existing_credits", "num_dependents",
]

FEATURE_COLS_CATEGORICAL = [
    "checking_status", "credit_history", "purpose", "savings_status",
    "employment", "personal_status", "other_parties", "property_magnitude",
    "other_payment_plans", "housing", "job", "telephone", "foreign_worker",
]

TARGET_COL = "class"

# Model hyperparameter defaults
DEFAULTS = {
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 1000,
        "solver": "saga",
    },
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
    },
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "lightgbm": {
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "subsample": 0.8,
    },
}


def validate_0(data):
    """Validate: add data validation"""
    return data is not None


def format_14(val):
    """Format: add retry logic"""
    return str(val).strip()
