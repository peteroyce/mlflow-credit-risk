"""
Training pipeline with MLflow experiment tracking.

Trains Logistic Regression, Random Forest, XGBoost, and LightGBM,
logging all params and metrics to MLflow.
"""

import logging

import mlflow
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.config import DEFAULTS, EXPERIMENT_NAME, MLFLOW_TRACKING_URI, RANDOM_SEED
from src.data.features import prepare_splits

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(
            **DEFAULTS["logistic_regression"], random_state=RANDOM_SEED,
        ),
        "random_forest": RandomForestClassifier(
            **DEFAULTS["random_forest"], random_state=RANDOM_SEED, n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            **DEFAULTS["xgboost"],
            random_state=RANDOM_SEED,
            eval_metric="logloss",
            use_label_encoder=False,
        ),
        "lightgbm": LGBMClassifier(
            **DEFAULTS["lightgbm"],
            random_state=RANDOM_SEED,
            verbose=-1,
        ),
    }


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def train_all():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test, preprocessor = prepare_splits()
    models = get_models()

    results = {}

    for name, model in models.items():
        logger.info("Training %s...", name)

        with mlflow.start_run(run_name=name):
            params = model.get_params()
            for k, v in params.items():
                if v is not None and not callable(v):
                    try:
                        mlflow.log_param(k, v)
                    except Exception:
                        pass

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics = compute_metrics(y_test, y_pred, y_prob)
            for k, v in metrics.items():
                mlflow.log_metric(k, round(v, 4))

            mlflow.sklearn.log_model(model, name)

            results[name] = metrics
            logger.info(
                "%s: accuracy=%.4f, f1=%.4f, roc_auc=%.4f",
                name, metrics["accuracy"], metrics["f1"], metrics["roc_auc"],
            )

    print("\n" + "=" * 65)
    print(f"{'Model':25s} {'Accuracy':>10s} {'F1':>8s} {'AUC-ROC':>10s}")
    print("-" * 65)
    for name, m in results.items():
        print(f"{name:25s} {m['accuracy']:10.4f} {m['f1']:8.4f} {m['roc_auc']:10.4f}")
    print("=" * 65)

    return results


if __name__ == "__main__":
    train_all()