"""
Model comparison and evaluation.

Loads trained models from MLflow, compares metrics, does threshold
tuning and SHAP feature importance analysis.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
import shap
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
)

from src.config import EXPERIMENT_NAME, MLFLOW_TRACKING_URI, RANDOM_SEED
from src.data.features import prepare_splits

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_best_model():
    """Load the best model (by AUC-ROC) from MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.roc_auc DESC"],
        max_results=1,
    )

    best_run = runs.iloc[0]
    logger.info(
        "Best model: %s (AUC-ROC: %.4f)",
        best_run["tags.mlflow.runName"],
        best_run["metrics.roc_auc"],
    )

    model_uri = f"runs:/{best_run.run_id}/{best_run['tags.mlflow.runName']}"
    model = mlflow.sklearn.load_model(model_uri)
    return model, best_run


def threshold_analysis(y_true, y_prob):
    """Find optimal threshold for different business objectives."""
    thresholds = np.arange(0.1, 0.9, 0.05)
    results = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    return results


def shap_analysis(model, X_test, feature_names=None):
    """Compute SHAP values for the best model."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:500])

    importance = np.abs(shap_values).mean(axis=0)
    if feature_names:
        return dict(sorted(
            zip(feature_names, importance),
            key=lambda x: -x[1],
        ))
    return importance


def evaluate():
    X_train, X_test, y_train, y_test, preprocessor = prepare_splits()
    model, best_run = load_best_model()

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report (best model):\n")
    print(classification_report(y_test, y_pred, target_names=["good", "bad"]))

    # Threshold analysis
    thresh_results = threshold_analysis(y_test, y_prob)
    best_f1_result = max(thresh_results, key=lambda x: x["f1"])
    print(f"Optimal threshold (by F1): {best_f1_result['threshold']:.2f}")
    print(f"  Precision: {best_f1_result['precision']:.4f}")
    print(f"  Recall:    {best_f1_result['recall']:.4f}")
    print(f"  F1:        {best_f1_result['f1']:.4f}")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve (AUC: {best_run['metrics.roc_auc']:.4f})")
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=150)
    plt.close()
    logger.info("ROC curve saved to roc_curve.png")


if __name__ == "__main__":
    evaluate()
