"""MCP stdio surface for Aurora's canonical runtime."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

from mcp.server.fastmcp import Context, FastMCP

from aurora.runtime.system import AuroraSystem, to_dict


@dataclass(frozen=True, slots=True)
class AuroraMCPContext:
    system: AuroraSystem


AuroraToolContext: TypeAlias = Context[Any, AuroraMCPContext, Any]


def _require_non_empty(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _resolve_data_dir() -> str | None:
    value = os.getenv("AURORA_DATA_DIR", "").strip()
    return value or None


@asynccontextmanager
async def _aurora_lifespan(_: FastMCP[AuroraMCPContext]) -> AsyncIterator[AuroraMCPContext]:
    system = AuroraSystem.create(data_dir=_resolve_data_dir())
    try:
        yield AuroraMCPContext(system=system)
    finally:
        system.close()


mcp = FastMCP("Aurora", lifespan=_aurora_lifespan)


def _system_from_context(ctx: AuroraToolContext) -> AuroraSystem:
    return ctx.request_context.lifespan_context.system


@mcp.tool(name="aurora_inject", structured_output=True)
def aurora_inject(
    payload: str,
    ctx: AuroraToolContext,
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
            _system_from_context(ctx).inject(
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
def aurora_read_workspace(
    cue: str,
    ctx: AuroraToolContext,
    session_id: str = "",
    k: int | None = None,
) -> dict[str, object]:
    return cast(
        dict[str, object],
        to_dict(
            _system_from_context(ctx).read_workspace(
                {"payload": _require_non_empty(cue, "cue"), "session_id": session_id},
                k=k,
            )
        ),
    )


@mcp.tool(name="aurora_maintenance_cycle", structured_output=True)
def aurora_maintenance_cycle(
    ctx: AuroraToolContext,
    ms_budget: int | None = None,
) -> dict[str, object]:
    return cast(dict[str, object], to_dict(_system_from_context(ctx).maintenance_cycle(ms_budget=ms_budget)))


@mcp.tool(name="aurora_respond", structured_output=True)
def aurora_respond(
    cue: str,
    ctx: AuroraToolContext,
    session_id: str = "",
    turn_id: str | None = None,
    source: str = "user",
    metadata: dict[str, object] | None = None,
    ts: int | None = None,
) -> dict[str, object]:
    return cast(
        dict[str, object],
        to_dict(
            _system_from_context(ctx).respond(
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
def aurora_snapshot(ctx: AuroraToolContext) -> dict[str, object]:
    return cast(dict[str, object], to_dict(_system_from_context(ctx).snapshot()))


@mcp.tool(name="aurora_field_stats", structured_output=True)
def aurora_field_stats(ctx: AuroraToolContext) -> dict[str, object]:
    return cast(dict[str, object], to_dict(_system_from_context(ctx).field_stats()))


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
