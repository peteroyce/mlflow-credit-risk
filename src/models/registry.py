"""
MLflow model registry helpers.

Handles registering the best model, transitioning between stages,
and loading the production model for serving.
"""

import logging

import mlflow
from mlflow.tracking import MlflowClient

from src.config import EXPERIMENT_NAME, MLFLOW_TRACKING_URI

logger = logging.getLogger(__name__)

MODEL_NAME = "credit-risk-classifier"


def get_client() -> MlflowClient:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()


def register_best_model() -> str:
    """Find the best run and register it in the model registry."""
    client = get_client()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.roc_auc DESC"],
        max_results=1,
    )

    best_run = runs.iloc[0]
    run_id = best_run.run_id
    model_name_in_run = best_run["tags.mlflow.runName"]

    model_uri = f"runs:/{run_id}/{model_name_in_run}"
    result = mlflow.register_model(model_uri, MODEL_NAME)

    logger.info(
        "Registered model '%s' version %s (run: %s, AUC: %.4f)",
        MODEL_NAME, result.version, run_id, best_run["metrics.roc_auc"],
    )
    return result.version


def promote_to_production(version: str):
    """Transition a model version to Production stage."""
    client = get_client()
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    logger.info("Model version %s promoted to Production", version)


def load_production_model():
    """Load the current Production model."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/Production"
    return mlflow.sklearn.load_model(model_uri)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    version = register_best_model()
    promote_to_production(version)