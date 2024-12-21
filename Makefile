.PHONY: check lint lint-checkonly type-check

check: lint type-check

lint: 
	uvx ruff check --fix
	uvx ruff format

lint-checkonly:
	uvx ruff check
	uvx ruff format --check

type-check:
	uv run pyright