# Engineering Notes

Aurora is archived. This repository keeps only the engineering surface needed to
understand, validate, and preserve the released system.

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

- `README.md` is the public engineering-facing overview
- `CONTRIBUTING.md` owns the contributor workflow
- `docs/` is reserved for current engineering documentation
- `tests/integration/` holds black-box coverage for published surfaces

The repository does not grow beyond those needs.
