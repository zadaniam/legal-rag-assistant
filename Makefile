# ==============================================================================
# Legal RAG Assistant — developer commands
# ==============================================================================
# Usage:
#   make                      # show this help
#   make up                   # start local Qdrant + PostgreSQL
#   make init-qdrant ingest   # initialise Qdrant and load mock data
#   make run-app              # run the backend  (http://localhost:7000)
#   make run-ui               # run the Chainlit UI (http://localhost:8000)
#
# Override the environment per command, e.g.:
#   make init-qdrant APP_ENV=production
# ==============================================================================

SHELL := /bin/bash
.DEFAULT_GOAL := help
MAKEFLAGS += --no-print-directory

# --- Configurable variables (override from the command line) ------------------
APP_ENV      ?= development
APP_PORT     ?= 7000
UI_PORT      ?= 8000
UV           ?= uv
COMPOSE      ?= docker compose
COMPOSE_FILE ?= docker-compose.dev.yml

PYTHON := $(UV) run python

.PHONY: help sync lint format pre-commit test test-unit test-integration \
        up down logs init-qdrant init-postgres ingest run-app run-ui \
        init-qdrant-prod ingest-qdrant-prod clean

# ==============================================================================
# Help
# ==============================================================================
help: ## Show this help message
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ==============================================================================
# Setup & code quality
# ==============================================================================
sync: ## Install all dependencies (including the dev group)
	$(UV) sync

lint: ## Lint and auto-fix with Ruff
	$(UV) run ruff check . --fix

format: ## Format code with Ruff
	$(UV) run ruff format .

types: ## Run static type checking with Mypy
	$(UV) run mypy .

pre-commit: ## Run all pre-commit hooks
	$(UV) run pre-commit run --all-files

test: test-unit test-integration ## Run the full unit & integration test suite

test-unit: ## Run unit tests only
	$(UV) run pytest tests/unit

test-nodes: ## Run nodes unit tests only
	$(UV) run pytest tests/unit/nodes

test-integration: ## Run integration tests only (requires the Docker databases)
	$(UV) run pytest tests/integration

test-e2e-api: ## Run e2e tests (requires running FastAPI and LLM API configuration)
	$(UV) run pytest tests/e2e/api

# ==============================================================================
# Local infrastructure
# ==============================================================================
up: ## Start local Qdrant + PostgreSQL (Docker Compose)
	$(COMPOSE) -f $(COMPOSE_FILE) up -d

down: ## Stop local Docker Compose services
	$(COMPOSE) -f $(COMPOSE_FILE) down

logs: ## Tail logs from the local services
	$(COMPOSE) -f $(COMPOSE_FILE) logs -f

# ==============================================================================
# Database initialisation & ingestion pipeline   (APP_ENV=$(APP_ENV))
# ==============================================================================
init-qdrant: ## Create the Qdrant collection
	APP_ENV=$(APP_ENV) $(PYTHON) -m src.database.init_qdrant

init-postgres: ## Create PostgreSQL tables
	APP_ENV=$(APP_ENV) $(PYTHON) -m src.database.init_postgres

data-preprocess : ## Preprocess raw legal documents into a format suitable for ingestion
	APP_ENV=$(APP_ENV) $(PYTHON) -m src.data_pipeline.preprocess.pipeline

data-ingest: ## Ingest preprocessed legal documents into Qdrant
	APP_ENV=$(APP_ENV) $(PYTHON) -m src.data_pipeline.ingestion.pipeline

# ==============================================================================
# Run the application
# ==============================================================================
run-app: ## Run the FastAPI backend (http://localhost:7000)
	APP_ENV=$(APP_ENV) $(UV) run uvicorn src.main:app --port $(APP_PORT) --reload

run-ui: ## Run the Chainlit UI (http://localhost:8000)
	APP_ENV=$(APP_ENV) CHAINLIT_APP_ROOT=ui $(UV) run python -m chainlit run ui/app_chainlit.py --port $(UI_PORT)

# ==============================================================================
# Production shortcuts (Configure the URLs in `.env.production` before running these)
# ==============================================================================
init-qdrant-prod: ## Shortcut: run `init-qdrant` with APP_ENV=production
	@$(MAKE) -f $(firstword $(MAKEFILE_LIST)) init-qdrant APP_ENV=production

data-ingest-prod: ## Shortcut: run `data-ingest` with APP_ENV=production
	@$(MAKE) -f $(firstword $(MAKEFILE_LIST)) data-ingest APP_ENV=production

# ==============================================================================
# Housekeeping
# ==============================================================================
clean: ## Remove Python caches
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache ui/.chainlit ui/.files htmlcov/ .coverage
