# Aurora

Aurora is now on a clean rewrite track.

This repository no longer centers on the old Seed v1 substrate stack.
It now centers on a new runtime model where Aurora has internal phases and continuity:

- `awake`
- `doze`
- `sleep`

The new goal is not a memory CRUD product.
The goal is a memory-bearing runtime that keeps internal state, forms relational history, and changes mostly internally before that change appears in behavior.

## Current Status

The rewrite track is active and running.

What is already live in code:

- new package structure (`being`, `memory`, `relation`, `phases`, `expression`, `persistence`, `runtime`, `surface`)
- new `AuroraEngine` runtime path
- canonical graph writes (`Turn`, `Fragment`, `Trace`, `Association`, `RelationMoment`, `Thread`, `Knot`, `RelationFormation`, `Orientation`, `MetabolicState`)
- phase operations: interaction turn, doze, sleep
- new HTTP surface endpoints and CLI commands
- strict type check and tests on the new track

What was removed:

- legacy `core_math` runtime path
- legacy `host_runtime` path
- legacy `substrate_core` path
- legacy `surface_api` path
- legacy memory CRUD API and related tests
- parallel `aurora_ontology_core` runtime track

## Philosophy Documents

Design intent and architecture decisions are documented in:

- `docs/aurora-architecture-principles.md`
- `docs/aurora-final-architecture-blueprint.md`
- `docs/aurora-module-map.md`
- `docs/aurora-rewrite-roadmap.md`
- `docs/aurora-api-contract.md`

Read these first before changing core behavior.

## Package Layout

```text
aurora/
├── being/
├── memory/
├── relation/
├── phases/
├── expression/
├── persistence/
├── runtime/
└── surface/
```

## Install

```bash
pip install -e .
```

Dev dependencies:

```bash
pip install -e '.[dev]'
```

## CLI

The CLI now uses the new runtime surface.

Run one interaction turn:

```bash
aurora turn "Hello Aurora"
```

Run doze:

```bash
aurora doze
```

Run sleep:

```bash
aurora sleep
```

## HTTP API

App entrypoint:

- `aurora/surface/api.py`

Current endpoints:

- `GET /health`
- `GET /state`
- `POST /turn`
- `POST /doze`
- `POST /sleep`

Example turn request:

```json
{
  "session_id": "default",
  "text": "I learned something important today"
}
```

## Quality Gates

Run tests:

```bash
uv run pytest -q
```

Run type checks:

```bash
uv run mypy aurora --show-error-codes --pretty
```

## Rewrite Policy

This repository follows a strict cleanup policy during rewrite:

- no compatibility theater
- no dead legacy execution path
- no giant god-files
- no stale framework baggage
- delete mismatched architecture instead of preserving it

If a module does not fit the new ontology, remove it.
