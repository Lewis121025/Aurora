"""MCP stdio surface for Aurora's canonical runtime."""

from __future__ import annotations

import atexit
from typing import cast

from mcp.server.fastmcp import FastMCP

from aurora.runtime.system import AuroraSystem, to_dict

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


@mcp.tool(name="aurora_inject", structured_output=True)
def aurora_inject(
    payload: str,
    session_id: str = "",
    turn_id: str | None = None,
    source: str = "user",
    payload_type: str = "text",
    metadata: dict[str, object] | None = None,
    ts: int | None = None,
) -> dict[str, object]:
    return cast(
        dict[str, object],
        to_dict(
        _get_system().inject(
            {
                "payload": _require_non_empty(payload, "payload"),
                "session_id": session_id,
                "turn_id": turn_id,
                "source": source,
                "payload_type": payload_type,
                "metadata": dict(metadata or {}),
                "ts": ts,
            }
        )
        ),
    )


@mcp.tool(name="aurora_read_workspace", structured_output=True)
def aurora_read_workspace(cue: str, session_id: str = "", k: int | None = None) -> dict[str, object]:
    return cast(
        dict[str, object],
        to_dict(
        _get_system().read_workspace(
            {"payload": _require_non_empty(cue, "cue"), "session_id": session_id},
            k=k,
        )
        ),
    )


@mcp.tool(name="aurora_maintenance_cycle", structured_output=True)
def aurora_maintenance_cycle(ms_budget: int | None = None) -> dict[str, object]:
    return cast(dict[str, object], to_dict(_get_system().maintenance_cycle(ms_budget=ms_budget)))


@mcp.tool(name="aurora_respond", structured_output=True)
def aurora_respond(
    cue: str,
    session_id: str = "",
    turn_id: str | None = None,
    source: str = "user",
    metadata: dict[str, object] | None = None,
    ts: int | None = None,
) -> dict[str, object]:
    return cast(
        dict[str, object],
        to_dict(
        _get_system().respond(
            {
                "payload": _require_non_empty(cue, "cue"),
                "session_id": session_id,
                "turn_id": turn_id,
                "source": source,
                "metadata": dict(metadata or {}),
                "ts": ts,
            }
        )
        ),
    )


@mcp.tool(name="aurora_snapshot", structured_output=True)
def aurora_snapshot() -> dict[str, object]:
    return cast(dict[str, object], to_dict(_get_system().snapshot()))


@mcp.tool(name="aurora_field_stats", structured_output=True)
def aurora_field_stats() -> dict[str, object]:
    return cast(dict[str, object], to_dict(_get_system().field_stats()))


def main() -> None:
    _get_system()
    try:
        mcp.run(transport="stdio")
    finally:
        _close_system()


if __name__ == "__main__":
    main()
