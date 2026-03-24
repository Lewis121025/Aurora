# Engineering Notes

Aurora is a single-user local deployment system. The repository keeps a compact
engineering surface on purpose: only assets that directly improve correctness,
reviewability, and maintenance belong here.

## Python Policy

- Minimum supported runtime: Python 3.13
- Blocking CI matrix: Python 3.13 and 3.14
- Forward smoke check: Python 3.15-dev, non-blocking

## Dependency Policy

- `pyproject.toml` defines the supported dependency floor
- `uv.lock` defines the exact tested environment
- Dependabot updates GitHub Actions and `uv` dependencies weekly
- Runtime and developer dependencies are grouped separately in Dependabot

## Repository Shape

- `README.md` stays user-facing
- `CONTRIBUTING.md` owns the contributor workflow
- `docs/` is reserved for current engineering or product documentation
- `tests/integration/` holds black-box happy-path coverage for published surfaces

Aurora does not mirror the full `mem0` repository shape. It only adopts the
engineering assets that solve an active maintenance problem in this repository.
