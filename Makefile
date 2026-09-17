.PHONY: help install lint format check test ingest serve evaluate debug-failures

help:
	@echo "Available commands:"
	@echo "  make install          Install project and development dependencies"
	@echo "  make lint             Run Ruff lint checks"
	@echo "  make format           Format source and test files with Ruff"
	@echo "  make check            Run lint, format, and tests"
	@echo "  make test             Run the pytest suite"
	@echo "  make ingest           Load PDFs, chunk them, embed them, and store them"
	@echo "  make serve            Start the FastAPI server"
	@echo "  make evaluate         Run the RAGAS evaluation"
	@echo "  make debug-failures   Inspect low-scoring evaluation samples"

install:
	uv sync

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

check: lint
	uv run ruff format --check src/ tests/
	uv run pytest tests/ -v --tb=short

test:
	uv run pytest tests/ -v --tb=short

ingest:
	uv run ingest

serve:
	uv run serve

evaluate:
	uv run evaluate

debug-failures:
	uv run python -m rag_chatbot.evaluation.debug_failures
