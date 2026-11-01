.PHONY: help install install-vision lint fmt typecheck test test-unit test-int test-regression \
        clean data migrate train train-all evaluate serve docker-build docker-run ci

PY ?= python
VENV := .venv
ifeq ($(OS),Windows_NT)
  BIN := $(VENV)/Scripts
else
  BIN := $(VENV)/bin
endif
PYBIN := $(BIN)/python

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Create the venv and install core + dev dependencies
	$(PY) -m venv $(VENV)
	$(PYBIN) -m pip install --upgrade pip
	$(PYBIN) -m pip install -e ".[dev]"

install-vision:  ## Add the optional torch/torchvision extra
	$(PYBIN) -m pip install -e ".[vision]"

lint:  ## Ruff lint + format check
	$(PYBIN) -m ruff check src tests
	$(PYBIN) -m ruff format --check src tests

fmt:  ## Auto-fix lint and format
	$(PYBIN) -m ruff check --fix src tests
	$(PYBIN) -m ruff format src tests

typecheck:  ## mypy over src/
	$(PYBIN) -m mypy

test:  ## Full suite except slow
	$(PYBIN) -m pytest -m "not slow"

test-unit:  ## Unit tests only
	$(PYBIN) -m pytest -m unit

test-int:  ## Integration tests
	$(PYBIN) -m pytest -m integration

test-regression:  ## Metric floors + frozen API contract
	$(PYBIN) -m pytest -m regression

data:  ## Validate every raw dataset against its declared schema
	$(PYBIN) -m agri data --all

train:  ## Train one module:  make train M=yield_prediction
	$(PYBIN) -m agri train --module $(M)

train-all:  ## Train every module in dependency order
	$(PYBIN) -m agri train --all

evaluate:  ## Evaluate one module:  make evaluate M=crop_recommendation
	$(PYBIN) -m agri evaluate --module $(M)

serve:  ## Run the API locally with reload
	$(PYBIN) -m uvicorn agri.api.app:app --reload --port 8000

docker-build:
	docker build -t smart-crop-ai:latest -f docker/Dockerfile .

docker-run:
	docker run --rm -p 8000:8000 --env-file .env smart-crop-ai:latest

ci: lint typecheck test  ## What CI runs

clean:  ## Remove caches and build output
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +