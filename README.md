# Aurora

Aurora is a unified trace-field memory system. Raw experience enters one evolving field, and replay, consolidation, competition, compression, forgetting, and workspace readout all happen inside that same runtime.

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

system.inject(
    {
        "payload": "I live in Hangzhou.",
        "session_id": "session-a",
        "turn_id": "turn-1",
        "source": "user",
    }
)
workspace = system.read_workspace({"payload": "Where do I live?", "session_id": "session-a"})
reply = system.respond({"payload": "What city do I live in?", "session_id": "session-a"})

system.close()
```

Public runtime surface:

- `AuroraSystem.create(...) -> AuroraSystem`
- `inject(raw_event) -> InjectResult`
- `maintenance_cycle(ms_budget=None) -> MaintenanceStats`
- `read_workspace(cue, k=None) -> Workspace`
- `respond(cue) -> ResponseResult`
- `snapshot() -> SnapshotMeta`
- `field_stats() -> FieldStats`
- `close() -> None`

## CLI

```bash
aurora inject --payload "I live in Hangzhou." --session-id session-a --turn-id turn-1 --source user
aurora read-workspace --cue "Where do I live?" --session-id session-a
aurora maintenance-cycle --ms-budget 12
aurora respond --cue "What city do I live in?" --session-id session-a
aurora snapshot
aurora field-stats
```

## MCP

```bash
aurora-mcp
```

Exposed MCP interface:

- tools: `aurora_inject`, `aurora_read_workspace`, `aurora_maintenance_cycle`, `aurora_respond`, `aurora_snapshot`, `aurora_field_stats`

## HTTP API

```bash
uv run aurora serve --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Description |
| --- | --- | --- |
| `/health` | `GET` | Health check |
| `/inject` | `POST` | Inject one raw event |
| `/read-workspace` | `POST` | Read a structured workspace |
| `/maintenance-cycle` | `POST` | Run one maintenance cycle |
| `/respond` | `POST` | Generate one response turn |
| `/snapshot` | `POST` | Persist an internal snapshot |
| `/field-stats` | `GET` | Read runtime statistics |

## Runtime Model

Each input follows the same write path:

1. Packetize raw input only by mechanical boundaries.
2. Persist raw payloads and anchors.
3. Propose `birth / assimilate / split / attach` against existing traces.
4. Let replay, procedure induction, prototype induction, and budget control reshape the same field over time.

The runtime is split into:

- `aurora/runtime`: canonical `AuroraField` and `AuroraSystem`
- `aurora/surfaces`: HTTP, CLI, and MCP transports
- `aurora/expression`: workspace rendering and response context
- `aurora/core`: canonical types, config, and math
- `aurora/store`: blob, trace, edge, ANN, and snapshot persistence

The implemented kernel follows the scientist spec at a practical v2 baseline:

- local-energy proposal over `ASSIMILATE / ATTACH / SPLIT / BIRTH`
- posterior groups with an explicit null slot
- sparse workspace settling with inhibitory edges
- replay-driven slow predictor updates
- prototype / procedure role promotion inside the same trace schema
- hard budget pruning inside the field rather than offline cleanup

## Validation

```bash
uv run pytest -q
uv run mypy aurora tests --show-error-codes --pretty
uv run ruff check aurora tests
python -m compileall -q aurora tests
```

## License

Proprietary
