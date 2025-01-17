.PHONY: check lint lint-checkonly type-check

check: lint type-check test

lint: 
	uvx ruff check --fix
	uvx ruff format

lint-checkonly:
	uvx ruff check
	uvx ruff format --check

type-check:
	uv run pyright

test:
	uv run pytest