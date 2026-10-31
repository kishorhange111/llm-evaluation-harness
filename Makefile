# Common developer commands. Run `make help` to list them.
.DEFAULT_GOAL := help
.PHONY: help install install-dev lint format test check clean smoke

help:           ## list the available commands
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-12s %s\n", $$1, $$2}'

install:        ## this package + local Hugging Face inference
	pip install -e ".[hf]"

install-dev:    ## lint/test tooling + this package (editable)
	pip install -e ".[dev]" transformers accelerate

lint:           ## ruff lint + format check (what CI runs)
	ruff check .
	ruff format --check .

format:         ## auto-fix lint findings and reformat
	ruff check --fix .
	ruff format .

test:           ## run the test suite (CPU only)
	pytest

check: lint test ## everything CI runs

smoke:          ## a two-minute end-to-end evaluation run
	evalkit run configs/smoke.yaml --limit 10

clean:          ## remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
