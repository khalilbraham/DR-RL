.DEFAULT_GOAL := help
UV ?= uv

.PHONY: help install lock fmt lint type test cov check smoke mve clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Sync the dev environment from the lockfile
	$(UV) sync --group dev

lock: ## Regenerate the lockfile
	$(UV) lock

fmt: ## Auto-format with ruff
	$(UV) run ruff format src tests
	$(UV) run ruff check --fix src tests

lint: ## Lint (ruff check + format check)
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

type: ## Static type-check (mypy --strict)
	$(UV) run mypy

test: ## Run the fast unit suite (no slow/gpu/integration)
	$(UV) run pytest -m "not slow and not gpu and not integration"

cov: ## Run tests with coverage gate
	$(UV) run pytest -m "not slow and not gpu and not integration" \
		--cov=drrl --cov-report=term-missing --cov-fail-under=85

check: lint type cov ## Phase-0 gate: lint -> type -> test+coverage

smoke: ## Compose config, seed, and write a run manifest (sanity entrypoint)
	$(UV) run python -m experiments.smoke

mve: ## Minimum Viable Experiment (available from Phase 5)
	@echo "MVE is wired up in Phase 5 (GRPO). See README 'Build order'."
	@exit 1

clean: ## Remove caches and build artifacts
	rm -rf .mypy_cache .ruff_cache .pytest_cache .hypothesis htmlcov \
		.coverage coverage.xml build dist *.egg-info
