# mlflow-credit-risk

Credit risk prediction pipeline with MLflow experiment tracking. Compares Logistic Regression, Random Forest, XGBoost, and LightGBM on the German Credit dataset.

## Results

| Model | Accuracy | F1 | AUC-ROC |
|-------|----------|-----|---------|
| Logistic Regression | 0.735 | 0.521 | 0.764 |
| Random Forest | 0.760 | 0.567 | 0.791 |
| XGBoost | 0.775 | 0.594 | 0.803 |
| **LightGBM** | **0.780** | **0.601** | **0.809** |

After Optuna tuning, XGBoost reaches **0.812 AUC-ROC**.

## How it works

1. **Data ingestion** - fetch German Credit dataset, validate, save locally
2. **Feature engineering** - StandardScaler + OneHotEncoder pipeline, derived features (amount_per_month, age bins, credit_load)
3. **Training** - 4 models trained with all params/metrics logged to MLflow
4. **Evaluation** - model comparison, threshold tuning, SHAP feature importance
5. **Registry** - best model registered and promoted to Production stage
6. **Serving** - FastAPI endpoint loads Production model from MLflow

## Setup

```bash
pip install -e .
```

## Usage

```bash
# download and prepare data
make ingest

# train all models (logs to MLflow)
make train

# compare models and analyze
make evaluate

# start MLflow UI
make mlflow-ui

# serve predictions
make serve
```

## Tech

MLflow, XGBoost, LightGBM, scikit-learn, Optuna, SHAP, FastAPI
