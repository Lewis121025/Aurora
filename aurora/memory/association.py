from __future__ import annotations

from dataclasses import dataclass

from aurora.runtime.contracts import AssocKind


@dataclass(frozen=True, slots=True)
class Association:
    edge_id: str
    src_fragment_id: str
    dst_fragment_id: str
    kind: AssocKind
    weight: float
    evidence: tuple[str, ...]
    created_at: float
    last_touched_at: float
