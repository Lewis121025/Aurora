from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from aurora.core.models.trace import QueryHit


@dataclass
class IngestResult:
    event_id: str
    plot_id: str
    story_id: Optional[str]
    memory_layer: Literal["explicit", "shadow"]
    tension: float
    surprise: float
    pred_error: float
    redundancy: float


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
    query_type: str
    attractor_path_len: int
    hit_count: int
    timeline_count: int
    standalone_count: int
    abstain: bool
    abstention_reason: str = ""
    asker_id: Optional[str] = None
    activated_identity: Optional[str] = None


@dataclass(frozen=True)
class StructuredMemoryContext:
    known_facts: List[str] = field(default_factory=list)
    preferences: List[str] = field(default_factory=list)
    relationship_state: List[str] = field(default_factory=list)
    active_narratives: List[str] = field(default_factory=list)
    temporal_context: List[str] = field(default_factory=list)
    system_intuition: List[str] = field(default_factory=list)
    cautions: List[str] = field(default_factory=list)
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


@dataclass
class CoherenceResult:
    overall_score: float
    conflict_count: int
    unfinished_story_count: int
    recommendations: List[str]
