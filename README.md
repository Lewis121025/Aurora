# Aurora

>
> Aurora is a first-principles adaptive memory runtime organized around one evolving trace field. Raw events, replay, consolidation, compression, forgetting, workspace readout, and response generation operate against the same runtime state instead of a separate retrieval stack.
>
> The repository targets single-user local deployment on Python 3.13+ and preserves four public surfaces: a Python SDK, a CLI, an HTTP API, and an MCP server. For the longer architecture reconstruction, see [Aurora.md](Aurora.md).

## What Aurora Provides

- A Python SDK centered on `AuroraSystem`
- A CLI exposed as `aurora`
- A FastAPI application surface for local HTTP integration
- An MCP stdio server exposed as `aurora-mcp`

## Requirements and Installation

- Python 3.13 or newer
- `uv`

Install the locked runtime environment:

```bash
uv sync --frozen
```

Aurora is designed for single-user local deployment.

## Quickstart

LLM settings are only required if you call `respond(...)` or use a surface that can generate responses.

### Python SDK

```python
from aurora import AuroraSystem

system = AuroraSystem.create(data_dir=".aurora-demo")

try:
    system.inject(
        {
            "payload": "I live in Hangzhou.",
            "session_id": "session-a",
            "turn_id": "turn-1",
            "source": "user",
        }
    )
    workspace = system.read_workspace(
        {"payload": "Where do I live?", "session_id": "session-a"}
    )
    reply = system.respond(
        {"payload": "What city do I live in?", "session_id": "session-a"}
    )

    print(workspace.active_trace_ids)
    print(reply.response_text)
finally:
    system.close()
```

`read_workspace(...)` is side-effect free. Runtime mutations happen through `inject(...)`, `maintenance_cycle(...)`, `respond(...)`, and `snapshot()`.

### CLI

Use a dedicated data directory in examples so the commands do not depend on any existing `.aurora` state in the current workspace.

```bash
uv run aurora --data-dir .aurora-demo inject \
  --payload "I live in Hangzhou." \
  --session-id session-a \
  --turn-id turn-1 \
  --source user

uv run aurora --data-dir .aurora-demo read-workspace \
  --cue "Where do I live?" \
  --session-id session-a

uv run aurora --data-dir .aurora-demo respond \
  --cue "What city do I live in?" \
  --session-id session-a
```

### HTTP API

```bash
uv run aurora --data-dir .aurora-demo serve --host 0.0.0.0 --port 8000
```

### MCP Server

Start the MCP server with an isolated data directory. This avoids inheriting any incompatible snapshot state from an existing `.aurora` directory in the current workspace.

```bash
AURORA_DATA_DIR=.aurora-mcp uv run aurora-mcp
```

## Configuration

Aurora reads the following environment variables:

| Variable | Required | Purpose |
| --- | --- | --- |
| `AURORA_LLM_PROVIDER` | For `respond(...)` only | LLM provider name |
| `AURORA_LLM_CONFIG_BASE_URL` | For `respond(...)` only | Provider base URL |
| `AURORA_LLM_CONFIG_MODEL` | For `respond(...)` only | Model identifier |
| `AURORA_LLM_CONFIG_API_KEY` | For `respond(...)` only | Provider API key |
| `AURORA_LLM_CONFIG_TIMEOUT_S` | Optional | Request timeout override |
| `AURORA_LLM_CONFIG_MAX_TOKENS` | Optional | Response token cap |
| `AURORA_API_KEY` | Optional | Bearer token for the HTTP API |
| `AURORA_DATA_DIR` | Optional | Runtime storage root for `aurora-mcp` |

Example configuration:

```env
AURORA_LLM_PROVIDER=openai
AURORA_LLM_CONFIG_BASE_URL=https://api.openai.com/v1
AURORA_LLM_CONFIG_MODEL=gpt-4o-mini
AURORA_LLM_CONFIG_API_KEY=
AURORA_LLM_CONFIG_TIMEOUT_S=30.0
AURORA_LLM_CONFIG_MAX_TOKENS=1024
AURORA_API_KEY=
AURORA_DATA_DIR=
```

When `AURORA_API_KEY` is set, every HTTP endpoint except `/health`, `/docs`, and `/openapi.json` requires `Authorization: Bearer ...`.

## Public Interfaces

### Python Package

Root exports:

- `AuroraSystem`
- `AuroraField`
- `AuroraSystemConfig`
- `FieldConfig`
- `build_app`

Primary `AuroraSystem` methods:

- `AuroraSystem.create(...) -> AuroraSystem`
- `inject(raw_event) -> InjectResult`
- `maintenance_cycle(ms_budget=None) -> MaintenanceStats`
- `read_workspace(cue, k=None) -> Workspace`
- `respond(cue) -> ResponseResult`
- `snapshot() -> SnapshotMeta`
- `field_stats() -> FieldStats`
- `close() -> None`

### CLI

The `aurora` CLI exposes the following subcommands:

- `inject`
- `read-workspace`
- `maintenance-cycle`
- `respond`
- `snapshot`
- `field-stats`
- `serve`

### HTTP API

| Endpoint | Method | Description |
| --- | --- | --- |
| `/health` | `GET` | Health check |
| `/inject` | `POST` | Inject one raw event |
| `/read-workspace` | `POST` | Read a structured workspace |
| `/maintenance-cycle` | `POST` | Run one maintenance cycle |
| `/respond` | `POST` | Generate one response turn |
| `/snapshot` | `POST` | Persist an internal snapshot |
| `/field-stats` | `GET` | Read runtime statistics |

### MCP Tools

The `aurora-mcp` server exposes the following tools:

- `aurora_inject`
- `aurora_read_workspace`
- `aurora_maintenance_cycle`
- `aurora_respond`
- `aurora_snapshot`
- `aurora_field_stats`

## Architecture Summary

Aurora treats memory as one evolving field rather than a pipeline of separate caches, summaries, and retrieval indexes.

- Raw input is packetized by mechanical boundaries and stored as anchored evidence.
- Traces are the mutable memory carriers; replay and maintenance can consolidate, split, reinterpret, or compress them over time.
- Workspace readout and response generation operate against the same field, so recall is a state readout problem rather than a separate compiled memory layer.

The public engineering surface in this repository is intentionally smaller than the full architecture discussion. For the longer first-principles reconstruction, see [aurora.md](aurora.md).

## Validation and Development

For the contributor workflow and release-safe validation gate, see [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
make sync
make check
uv run pre-commit run --all-files
```

## License

MIT. See [LICENSE](LICENSE).
