# Aurora

Aurora is a single-subject memory kernel. It stores immutable memory atoms, immutable edges, and a derived activation cache. State and recall are read-only projections over the current memory field.

## Installation

```bash
pip install -e '.[dev]'
```

## Configuration

Set the LLM configuration in `.env` or pass the same shape directly to `AuroraKernel.create(...)`.

```env
AURORA_LLM_PROVIDER=openai
AURORA_LLM_CONFIG_BASE_URL=https://api.openai.com/v1
AURORA_LLM_CONFIG_MODEL=gpt-4o-mini
AURORA_LLM_CONFIG_API_KEY=your-api-key
AURORA_API_KEY=your-http-api-key
```

`AURORA_API_KEY` is optional. When it is set, every HTTP endpoint except `/health`, `/docs`, and `/openapi.json` requires `Authorization: Bearer ...`.

Aurora exposes one public LLM settings shape:

```python
{
    "provider": "...",
    "config": {
        "base_url": "...",
        "model": "...",
        "api_key": "...",
    },
}
```

The runtime accepts providers backed by the OpenAI-compatible chat completions protocol.

## Python SDK

```python
from aurora.runtime.engine import AuroraKernel

kernel = AuroraKernel.create(
    llm_settings={
        "provider": "openai",
        "config": {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
            "api_key": "your-api-key",
        },
    }
)

turn = kernel.turn("subject-alice", "I live in Hangzhou and I like jazz.")
state = kernel.state("subject-alice")
recall = kernel.recall("subject-alice", "Hangzhou jazz", limit=8)

kernel.close()
```

If `llm_settings` is omitted, `AuroraKernel.create()` reads `AURORA_LLM_PROVIDER` and `AURORA_LLM_CONFIG_*`.

Public methods:

- `turn(subject_id, text, now_ts=None) -> TurnOutput`
- `state(subject_id) -> SubjectMemoryState`
- `recall(subject_id, query, limit=8) -> RecallResult`
- `close() -> None`

## CLI

```bash
aurora turn "Hello Aurora" --subject-id subject-alice
aurora state --subject-id subject-alice
aurora recall "Where do I live?" --subject-id subject-alice --limit 8
aurora status
```

## MCP

```bash
aurora-mcp
```

Exposed MCP interface:

- tools: `aurora_turn`, `aurora_recall`
- resources: `aurora://subject/{subject_id}/memory-field`

## HTTP API

```bash
uv run uvicorn aurora.surface.api:create_app --factory --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Body | Description |
| --- | --- | --- | --- |
| `/health` | `GET` | - | Health check |
| `/turn` | `POST` | `{"subject_id": "...", "text": "...", "now_ts": 0}` | Execute one turn |
| `/state/{subject_id}` | `GET` | - | Read the current memory field |
| `/recall` | `POST` | `{"subject_id": "...", "query": "...", "limit": 8}` | Read a query-scoped memory slice |

## Runtime Model

Each turn follows the same flow:

1. Append a user evidence atom.
2. Compile the input into new `memory`, `episode`, and `inhibition` nodes plus signed edges.
3. Evolve the subject memory field and refresh the activation cache.
4. Build a `MemoryBrief` from state and recall.
5. Generate a response.
6. Append an assistant evidence atom.
7. Compile the completed exchange into new nodes and edges.
8. Evolve the field again.

Aurora never mutates old atoms. Conflict, inhibition, and continuity are expressed through new immutable nodes and edges.

## Invariants

Aurora maintains these runtime boundaries:

1. Truth consists of immutable `memory_atoms`, immutable `memory_edges`, and derived `activation_cache`.
2. `state()` and `recall()` are read-only projections, not truth declarations.
3. Positive edges only propagate activation above intrinsic baseline.
4. Negative edges apply suppression pressure and do not encode logical negation.
5. `evidence` records observations only and does not participate in field coupling.
6. Payload decoding is strict. Invalid kinds, malformed payloads, and illegal references fail explicitly.

## Validation

The repository keeps unit tests and standard static checks only.

```bash
uv run pytest -q
uv run mypy aurora tests --show-error-codes --pretty
uv run ruff check aurora tests
python -m compileall -q aurora tests
```

## Project Layout

```text
aurora/
├── __main__.py
├── expression/
├── llm/
├── memory/
├── pipelines/
├── runtime/
└── surface/
```

## License

Proprietary
