.PHONY: sync test lint typecheck compile check

sync:
	uv sync --frozen --extra dev

test:
	uv run pytest -q

lint:
	uv run ruff check aurora tests

typecheck:
	uv run mypy aurora tests --show-error-codes --pretty

compile:
	uv run python -m compileall -q aurora tests

check: compile lint typecheck test
