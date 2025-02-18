.PHONY: check lint lint-checkonly type-check

check: lint type-check test-with-coverage

lint: 
	uvx ruff check --fix
	uvx ruff format

lint-checkonly:
	uvx ruff check
	uvx ruff format --check

type-check:
	uv run pyright

test:
	uv run pytest tests/

test-with-coverage:
	uv run pytest --cov-report term --cov src tests/