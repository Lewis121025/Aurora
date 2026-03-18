"""Aurora MCP stdio surface."""

from __future__ import annotations

import atexit
import json
from dataclasses import asdict

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from aurora.runtime.contracts import ActivatedAtom, ActivatedEdge, RecallResult, SubjectMemoryState, TurnOutput
from aurora.runtime.engine import AuroraKernel

mcp = FastMCP("Aurora")
_kernel: AuroraKernel | None = None
_cleanup_registered = False


class MCPTurnOutput(BaseModel):
    turn_id: str
    subject_id: str
    response_text: str
    recall_used: bool
    created_atom_ids: list[str] = Field(default_factory=list)
    created_edge_ids: list[str] = Field(default_factory=list)


class MCPActivatedAtom(BaseModel):
    atom_id: str
    atom_kind: str
    text: str
    activation: float
    confidence: float
    salience: float
    created_at: float


class MCPActivatedEdge(BaseModel):
    source_atom_id: str
    target_atom_id: str
    influence: float
    confidence: float


class MCPRecallOutput(BaseModel):
    subject_id: str
    query: str
    summary: str
    atoms: list[MCPActivatedAtom]
    edges: list[MCPActivatedEdge]


def _require_non_empty(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _require_positive_limit(limit: int) -> int:
    if limit < 1:
        raise ValueError("limit must be >= 1")
    return limit


def _close_kernel() -> None:
    global _kernel
    if _kernel is None:
        return
    _kernel.close()
    _kernel = None


def _get_kernel() -> AuroraKernel:
    global _kernel, _cleanup_registered
    if _kernel is None:
        _kernel = AuroraKernel.create()
        if not _cleanup_registered:
            atexit.register(_close_kernel)
            _cleanup_registered = True
    return _kernel


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _turn_payload(output: TurnOutput) -> MCPTurnOutput:
    return MCPTurnOutput(
        turn_id=output.turn_id,
        subject_id=output.subject_id,
        response_text=output.response_text,
        recall_used=output.recall_used,
        created_atom_ids=list(output.created_atom_ids),
        created_edge_ids=list(output.created_edge_ids),
    )


def _atom_payload(atom: ActivatedAtom) -> MCPActivatedAtom:
    return MCPActivatedAtom(**asdict(atom))


def _edge_payload(edge: ActivatedEdge) -> MCPActivatedEdge:
    return MCPActivatedEdge(**asdict(edge))


def _recall_payload(result: RecallResult) -> MCPRecallOutput:
    return MCPRecallOutput(
        subject_id=result.subject_id,
        query=result.query,
        summary=result.summary,
        atoms=[_atom_payload(atom) for atom in result.atoms],
        edges=[_edge_payload(edge) for edge in result.edges],
    )


def _state_payload(subject_id: str) -> SubjectMemoryState:
    return _get_kernel().state(_require_non_empty(subject_id, "subject_id"))


@mcp.tool(name="aurora_turn", structured_output=True)
def aurora_turn(subject_id: str, text: str, now_ts: float | None = None) -> MCPTurnOutput:
    """Run one subject-scoped turn."""
    return _turn_payload(
        _get_kernel().turn(
            subject_id=_require_non_empty(subject_id, "subject_id"),
            text=_require_non_empty(text, "text"),
            now_ts=now_ts,
        )
    )


@mcp.tool(name="aurora_recall", structured_output=True)
def aurora_recall(subject_id: str, query: str, limit: int = 8) -> MCPRecallOutput:
    """Recall one subject-scoped memory-field slice."""
    return _recall_payload(
        _get_kernel().recall(
            _require_non_empty(subject_id, "subject_id"),
            _require_non_empty(query, "query"),
            limit=_require_positive_limit(limit),
        )
    )


@mcp.resource(
    "aurora://subject/{subject_id}/memory-field",
    name="aurora_subject_memory_field",
    mime_type="application/json",
)
def subject_state(subject_id: str) -> str:
    """Get the current subject memory field."""
    return _json(asdict(_state_payload(subject_id)))


def main() -> None:
    """Run the MCP server over stdio."""
    _get_kernel()
    try:
        mcp.run(transport="stdio")
    finally:
        _close_kernel()


if __name__ == "__main__":
    main()
