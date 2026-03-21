# Aurora

Aurora is a unified ingest-and-evolve memory kernel. Every input becomes immutable memory material first. Reinforcement, suppression, replay, abstraction, and current-state readout all happen inside the same evolving field.

## Installation

```bash
pip install -e '.[dev]'
```

## Configuration

Aurora only needs LLM settings when you call `respond(...)` or serve an endpoint that can respond.

```env
AURORA_LLM_PROVIDER=openai
AURORA_LLM_CONFIG_BASE_URL=https://api.openai.com/v1
AURORA_LLM_CONFIG_MODEL=gpt-4o-mini
AURORA_LLM_CONFIG_API_KEY=your-api-key
AURORA_API_KEY=your-http-api-key
```

`AURORA_API_KEY` is optional. When it is set, every HTTP endpoint except `/health`, `/docs`, and `/openapi.json` requires `Authorization: Bearer ...`.

Public LLM settings shape:

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

## Python SDK

```python
from aurora import AuroraSystem

system = AuroraSystem.create()

system.ingest("I live in Hangzhou.", metadata={"speaker": "user"})
state = system.current_state()
recall = system.retrieve("Where do I live?")
reply = system.respond("session-a", "What city do I live in?")

system.close()
```

Public runtime surface:

- `AuroraSystem.create(...) -> AuroraSystem`
- `ingest(text, metadata=None, source="dialogue", now_ts=None) -> EventIngestResult`
- `ingest_batch(events, source="dialogue") -> dict`
- `retrieve(cue, top_k=8, propagation_steps=3) -> RecallResult`
- `current_state(top_k=10) -> RecallResult`
- `replay(budget=8, reason="replay") -> dict`
- `respond(session_id, text, metadata=None, source="dialogue", top_k=8, propagation_steps=3, now_ts=None) -> ResponseOutput`
- `stats() -> dict`
- `operation_history(limit=50) -> list[dict]`
- `get_atom(atom_id) -> dict`
- `close() -> None`

## CLI

```bash
aurora ingest --text "I live in Hangzhou." --metadata '{"speaker":"user"}'
aurora retrieve --cue "Where do I live?"
aurora current-state
aurora respond --session-id session-a --text "What city do I live in?"
aurora stats
```

## MCP

```bash
aurora-mcp
```

Exposed MCP interface:

- tools: `aurora_ingest`, `aurora_retrieve`, `aurora_current_state`, `aurora_replay`, `aurora_respond`
- resource: `aurora://memory/current-state`

## HTTP API

```bash
uv run uvicorn aurora.api:create_app --factory --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Description |
| --- | --- | --- |
| `/health` | `GET` | Health check |
| `/ingest` | `POST` | Ingest one event |
| `/ingest-batch` | `POST` | Ingest multiple events |
| `/retrieve` | `POST` | Query the evolving field |
| `/current-state` | `POST` | Read the current field projection |
| `/replay` | `POST` | Run replay over the field |
| `/respond` | `POST` | Generate a reply with short-lived session continuity |
| `/stats` | `GET` | Read system statistics |
| `/operations` | `GET` | Read operation history |
| `/atoms/{atom_id}` | `GET` | Inspect one atom and its edges |

## Runtime Model

Each input follows the same write path:

1. Store one raw anchor atom.
2. Compile one or more fact atoms from the input.
3. Link new atoms into the field with support, suppression, contradiction, and reference edges.
4. Let retrieval and replay reweight the same field over time instead of building separate summaries.

`retrieve()` and `current_state()` are stateful. Recall is part of reconsolidation, not a read-only projection.

## Validation

```bash
uv run pytest -q
uv run mypy aurora tests --show-error-codes --pretty
uv run ruff check aurora tests
python -m compileall -q aurora tests
```

## License

Proprietary
