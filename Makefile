.PHONY: install dev test lint clean download preprocess features train evaluate export preview

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

lint:
	ruff check lilly/ scripts/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf *.egg-info build dist .pytest_cache

# Data pipeline
download:
	python scripts/download.py

preprocess:
	python scripts/preprocess.py --workers 8

features:
	python scripts/extract_features.py

# Training
train-v1:
	python scripts/train.py --version v1

train-v2:
	python scripts/train.py --version v2

# Evaluation
evaluate-v1:
	python scripts/evaluate.py $(MODEL)

evaluate-v2:
	python scripts/evaluate.py --version v2 $(MODEL)

# Export
export-v1:
	python scripts/export.py $(MODEL) --quantize uint8

export-v2:
	python scripts/export.py --version v2 $(MODEL) --quantize uint8

# Preview
preview:
	python scripts/live_preview.py

pipeline: download preprocess features train-v1
