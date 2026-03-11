"""Runtime result models for Aurora V6."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from aurora.soul.models import IdentitySnapshot, Message, NarrativeSummary


MemoryKind = Literal["event", "plot", "summary", "story", "theme"]


@dataclass(frozen=True)
class PersistenceReceipt:
    event_id: str
    job_id: str
    status: Literal["accepted"]
    accepted_at: float
    projection_status: Literal["accepted", "projecting", "projected", "failed"]


@dataclass(frozen=True)
class QueryHit:
    id: str
    kind: MemoryKind
    score: float
    snippet: str
    metadata: Optional[Dict[str, str]] = None


@dataclass(frozen=True)
class QueryResult:
    query: str
    attractor_path_len: int
    overlay_hit_count: int
    hits: List[QueryHit]


@dataclass(frozen=True)
class EvidenceRef:
    id: str
    kind: MemoryKind
    score: float
    role: str


@dataclass(frozen=True)
class RetrievalTraceSummary:
    query: str
    attractor_path_len: int
    hit_count: int
    overlay_hit_count: int = 0
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
    overlay_hits: List[str] = field(default_factory=list)
    evidence_refs: List[EvidenceRef] = field(default_factory=list)


@dataclass(frozen=True)
class ChatTimings:
    retrieval_ms: float
    generation_ms: float
    persist_ms: float
    total_ms: float


@dataclass(frozen=True)
class ChatTurnResult:
    reply_message: Message
    event_id: str
    memory_context: StructuredMemoryContext
    rendered_memory_brief: str
    retrieval_trace_summary: RetrievalTraceSummary
    persistence: PersistenceReceipt
    timings: ChatTimings
    llm_error: Optional[str] = None


@dataclass(frozen=True)
class ChatStreamEvent:
    kind: Literal["status", "reply_delta", "done"]
    stage: Literal["retrieval", "generation", "persist_accept", "done"]
    text: str = ""
    result: Optional[ChatTurnResult] = None
