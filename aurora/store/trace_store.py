"""In-memory trace store."""

from __future__ import annotations

from aurora.core.types import TraceRecord


class TraceStore:
    """Persist evolving traces in memory for the field runtime."""

    def __init__(self) -> None:
        self.traces: dict[str, TraceRecord] = {}

    def add(self, trace: TraceRecord) -> None:
        self.traces[trace.trace_id] = trace

    def get(self, trace_id: str) -> TraceRecord | None:
        return self.traces.get(trace_id)

    def values(self) -> list[TraceRecord]:
        return list(self.traces.values())

    def remove(self, trace_id: str) -> None:
        self.traces.pop(trace_id, None)
