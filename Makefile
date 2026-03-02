.PHONY: install dev test lint clean download preprocess segment train evaluate export preview pipeline

# ── Setup ────────────────────────────────────────────────────────────────────

install:
	pip install -e .

dev:
	pip install -e ".[all]"

# ── Quality ──────────────────────────────────────────────────────────────────

test:
	python -m pytest tests/ -v

lint:
	ruff check lilly/ scripts/ tests/

lint-fix:
	ruff check --fix lilly/ scripts/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf *.egg-info build dist .pytest_cache htmlcov .coverage

# ── Data Pipeline ────────────────────────────────────────────────────────────

download:
	python scripts/download.py

preprocess:
	python scripts/preprocess.py --workers 8

segment:
	python scripts/segment_v3.py --workers 8

# ── Training ─────────────────────────────────────────────────────────────────

train:
	python scripts/train.py --epochs 50

# ── Evaluation ───────────────────────────────────────────────────────────────

evaluate:
	python scripts/evaluate.py $(MODEL) --tier 1

evaluate-full:
	python scripts/evaluate.py $(MODEL) --tier 1
	python scripts/evaluate.py $(MODEL) --tier 2 --n-samples 500
	python scripts/evaluate.py $(MODEL) --tier 3

# ── Inference ────────────────────────────────────────────────────────────────

preview:
	python scripts/live_preview.py

# ── Export ───────────────────────────────────────────────────────────────────

export:
	python scripts/export.py $(MODEL) --quantize uint8

# ── Full Pipeline ────────────────────────────────────────────────────────────

pipeline: download preprocess segment train
