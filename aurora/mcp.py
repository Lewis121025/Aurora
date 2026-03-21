"""MCP stdio surface for AuroraSystem."""

from __future__ import annotations

import atexit
import json

from mcp.server.fastmcp import FastMCP

from aurora.system import AuroraSystem, event_ingest_to_dict, recall_result_to_dict, response_output_to_dict

mcp = FastMCP("Aurora")
_system: AuroraSystem | None = None
_cleanup_registered = False


def _require_non_empty(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _close_system() -> None:
    global _system
    if _system is None:
        return
    _system.close()
    _system = None


def _get_system() -> AuroraSystem:
    global _system, _cleanup_registered
    if _system is None:
        _system = AuroraSystem.create()
        if not _cleanup_registered:
            atexit.register(_close_system)
            _cleanup_registered = True
    return _system


@mcp.tool(name="aurora_ingest", structured_output=True)
def aurora_ingest(text: str, metadata: dict[str, object] | None = None, source: str = "dialogue") -> dict[str, object]:
    return event_ingest_to_dict(
        _get_system().ingest(_require_non_empty(text, "text"), metadata=dict(metadata or {}), source=source)
    )


@mcp.tool(name="aurora_retrieve", structured_output=True)
def aurora_retrieve(cue: str, top_k: int = 8, propagation_steps: int = 3) -> dict[str, object]:
    return recall_result_to_dict(
        _get_system().retrieve(
            _require_non_empty(cue, "cue"),
            top_k=top_k,
            propagation_steps=propagation_steps,
        )
    )


@mcp.tool(name="aurora_current_state", structured_output=True)
def aurora_current_state(top_k: int = 10) -> dict[str, object]:
    return recall_result_to_dict(_get_system().current_state(top_k=top_k))


@mcp.tool(name="aurora_replay", structured_output=True)
def aurora_replay(budget: int = 8) -> dict[str, object]:
    return _get_system().replay(budget=budget)


@mcp.tool(name="aurora_respond", structured_output=True)
def aurora_respond(
    session_id: str,
    text: str,
    metadata: dict[str, object] | None = None,
    source: str = "dialogue",
    top_k: int = 8,
    propagation_steps: int = 3,
) -> dict[str, object]:
    return response_output_to_dict(
        _get_system().respond(
            _require_non_empty(session_id, "session_id"),
            _require_non_empty(text, "text"),
            metadata=dict(metadata or {}),
            source=source,
            top_k=top_k,
            propagation_steps=propagation_steps,
        )
    )


@mcp.resource(
    "aurora://memory/current-state",
    name="aurora_current_state",
    mime_type="application/json",
)
def current_state_resource() -> str:
    return json.dumps(recall_result_to_dict(_get_system().current_state()), ensure_ascii=False, indent=2)


def main() -> None:
    _get_system()
    try:
        mcp.run(transport="stdio")
    finally:
        _close_system()


if __name__ == "__main__":
    main()
