"""In-memory edge store."""

from __future__ import annotations

from aurora.core.types import TraceEdge


class EdgeStore:
    """Persist weighted couplings between traces."""

    def __init__(self) -> None:
        self.edges: dict[tuple[str, str, str], TraceEdge] = {}

    def upsert(self, edge: TraceEdge) -> None:
        self.edges[(edge.src, edge.dst, edge.kind)] = edge

    def outgoing(self, trace_id: str) -> list[TraceEdge]:
        return [edge for edge in self.edges.values() if edge.src == trace_id]

    def values(self) -> list[TraceEdge]:
        return list(self.edges.values())

    def remove(self, key: tuple[str, str, str]) -> None:
        self.edges.pop(key, None)

    def remove_trace(self, trace_id: str) -> None:
        for key in list(self.edges):
            edge = self.edges[key]
            if edge.src == trace_id or edge.dst == trace_id:
                del self.edges[key]
