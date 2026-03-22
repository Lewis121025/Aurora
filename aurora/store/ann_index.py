"""ANN index abstraction with an exact-search baseline backend."""

from __future__ import annotations

from aurora.core.math import cosine_similarity
from aurora.core.types import TraceRecord


class ExactANNIndex:
    """Brute-force nearest-neighbor search over trace means."""

    def __init__(self) -> None:
        self._vectors: dict[str, list[float]] = {}

    def add_or_update(self, trace: TraceRecord) -> None:
        self._vectors[trace.trace_id] = trace.z_mu.tolist()

    def remove(self, trace_id: str) -> None:
        self._vectors.pop(trace_id, None)

    def search(self, vector: list[float], *, top_k: int = 16) -> list[str]:
        ranked = sorted(
            ((cosine_similarity(vector, candidate), trace_id) for trace_id, candidate in self._vectors.items()),
            key=lambda item: (-item[0], item[1]),
        )
        return [trace_id for _, trace_id in ranked[:top_k]]
