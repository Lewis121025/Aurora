from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from aurora.soul.models import IdentitySnapshot, NarrativeSummary


@dataclass
class IngestResult:
    event_id: str
    plot_id: str
    story_id: Optional[str]
    mode: str
    source: Literal["wake", "dream", "repair", "mode"]
    tension: float
    contradiction: float
    active_energy: float
    repressed_energy: float


@dataclass(frozen=True)
class QueryHit:
    id: str
    kind: str
    score: float
    snippet: str
    metadata: Optional[Dict[str, str]] = None


@dataclass
class QueryResult:
    query: str
    attractor_path_len: int
    hits: List[QueryHit]


@dataclass(frozen=True)
class EvidenceRef:
    id: str
    kind: str
    score: float
    role: str


@dataclass(frozen=True)
class RetrievalTraceSummary:
    query: str
    attractor_path_len: int
    hit_count: int
    ranked_kinds: List[str] = field(default_factory=list)
    query_type: Optional[str] = None
    time_relation: Optional[str] = None
    time_start: Optional[float] = None
    time_end: Optional[float] = None
    time_anchor_event: Optional[str] = None


@dataclass(frozen=True)
class StructuredMemoryContext:
    mode: str
    narrative_pressure: float
    intuition: List[str] = field(default_factory=list)
    identity: Optional[IdentitySnapshot] = None
    narrative_summary: Optional[NarrativeSummary] = None
    retrieval_hits: List[str] = field(default_factory=list)
    evidence_refs: List[EvidenceRef] = field(default_factory=list)


@dataclass(frozen=True)
class ChatTimings:
    retrieval_ms: float
    generation_ms: float
    ingest_ms: float
    total_ms: float


@dataclass(frozen=True)
class ChatTurnResult:
    reply: str
    event_id: str
    memory_context: StructuredMemoryContext
    rendered_memory_brief: str
    system_prompt: str
    user_prompt: str
    retrieval_trace_summary: RetrievalTraceSummary
    ingest_result: IngestResult
    timings: ChatTimings
    llm_error: Optional[str] = None
