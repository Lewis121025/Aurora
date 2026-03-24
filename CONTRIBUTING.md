# Contributing

Aurora is archived. Keep changes minimal, correct, and release-safe.

Aurora targets single-user local deployment and supports Python 3.13 or newer.
The supported validation matrix is Python 3.13 and 3.14, with Python 3.15-dev
running as a non-blocking smoke check in CI.

## Prerequisites

- Python 3.13 or newer
- `uv`
- `make`

## Local Workflow

```bash
make sync
```

`make sync` runs `uv sync --frozen --extra dev` and installs the locked
development environment from `uv.lock`. Do not use `pip install -e` in this
repository.

## Validation

Run the full local gate before opening or updating a pull request:

```bash
make check
uv run pre-commit run --all-files
```

The fixed developer entrypoints are:

```bash
make sync
make test
make lint
make typecheck
make compile
make check
```

## Repository Rules

- Keep the runtime public API stable unless the change explicitly targets it.
- Keep docs and integration tests in the repository when they document or verify
  current behavior.
- Do not commit generated artifacts, local databases, virtual environments,
  caches, or exports.
- Treat `pyproject.toml` and `uv.lock` as the single dependency contract.

## CI Expectations

- Blocking validation runs on Python 3.13 and 3.14.
- Python 3.15-dev is a non-blocking forward smoke check.
- Dependabot updates GitHub Actions and `uv` dependencies weekly.
