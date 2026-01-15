.PHONY: setup ingest train evaluate serve test clean mlflow-ui

setup:
	pip install -e ".[dev]"

ingest:
	python -m src.data.ingest

train:
	python -m src.models.train

evaluate:
	python -m src.models.evaluate

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v

mlflow-ui:
	mlflow ui --port 5000

clean:
	rm -rf mlruns/ mlartifacts/ __pycache__ .ipynb_checkpoints
	find . -name "*.pyc" -delete
