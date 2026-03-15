# Aurora

Aurora is a memory-bearing runtime with awake/doze/sleep continuity.

It keeps internal state, forms relational history, and changes mostly internally before that change appears in behavior. When an LLM is configured, Aurora generates language from graph-derived context. Without one, it falls back to template rendering.

## Package Layout

```text
aurora/
├── being/          # MetabolicState, Orientation, touch detection
├── evaluation/     # ontology regression checks
├── expression/     # response planning, rendering, LLM prompt assembly
├── llm/            # LLMProvider protocol, OpenAI-compatible implementation
├── memory/         # Fragment, Trace, Association, Thread, Knot, recall, reweave
├── persistence/    # SQLite persistence (UPSERT-based)
├── phases/         # awake, doze, sleep lifecycle orchestration
├── relation/       # RelationMoment, RelationFormation, projectors
├── runtime/        # AuroraEngine, contracts, state, projections
└── surface/        # HTTP API, CLI
```

## Install

```bash
pip install -e '.[dev]'
```

## LLM Configuration

Copy `.env.example` to `.env` and set:

```
AURORA_LLM_BASE_URL=https://api.openai.com/v1
AURORA_LLM_MODEL=gpt-4o-mini
AURORA_LLM_API_KEY=your-key
```

Any OpenAI-compatible API works (Bailian, DeepSeek, etc.). Without these variables, Aurora uses template rendering.

## CLI

```bash
aurora turn "Hello Aurora"
aurora doze
aurora sleep
```

## HTTP API

Endpoints: `GET /health`, `GET /state`, `POST /turn`, `POST /doze`, `POST /sleep`

```json
{"session_id": "default", "text": "I learned something important today"}
```

## Quality Gates

```bash
uv run pytest -q
uv run mypy aurora --show-error-codes --pretty
uv run ruff check aurora
```

## Architecture Documents

- `docs/aurora-architecture-principles.md` — north star
- `docs/aurora-final-architecture-blueprint.md` — target system shape
- `docs/aurora-module-map.md` — active module boundaries
- `docs/aurora-rewrite-roadmap.md` — completed and remaining work
- `docs/aurora-api-contract.md` — HTTP contract
