from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "v5"


class MemoryKind(str, Enum):
    EVENT = "event"
    PLOT = "plot"
    SUMMARY = "summary"
    STORY = "story"
    THEME = "theme"


class AcceptedInteractionRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    event_id: Optional[str] = None
    session_id: str = "default"
    user_message: str = Field(..., min_length=1)
    agent_message: str = Field(..., min_length=1)
    actors: Optional[List[str]] = None
    context: Optional[str] = None
    ts: Optional[float] = None


class QueryRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    text: str = Field(..., min_length=1)
    k: int = Field(default=8, ge=1, le=50)
    session_id: Optional[str] = None


class ReplyRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    session_id: str = "default"
    user_message: str = Field(..., min_length=1)
    event_id: Optional[str] = None
    context: Optional[str] = None
    actors: Optional[List[str]] = None
    k: int = Field(default=6, ge=1, le=20)
    ts: Optional[float] = None


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    query_text: str
    chosen_id: str
    success: bool


class PersistenceReceiptResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    event_id: str
    job_id: str
    status: Literal["accepted"]
    accepted_at: float
    projection_status: Literal["accepted", "projecting", "projected", "failed"]


class QueryHitResponse(BaseModel):
    id: str
    kind: MemoryKind
    score: float
    snippet: str
    metadata: Optional[Dict[str, str]] = None


class QueryResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    query: str
    attractor_path_len: int
    overlay_hit_count: int
    hits: List[QueryHitResponse]


class EvidenceRef(BaseModel):
    id: str
    kind: MemoryKind
    score: float
    role: str


class IdentitySnapshotResponse(BaseModel):
    current_mode: str
    axis_state: Dict[str, float]
    intuition_axes: Dict[str, float]
    persona_axes: Dict[str, Dict[str, object]]
    axis_aliases: Dict[str, str]
    modes: Dict[str, Dict[str, object]]
    active_energy: float
    repressed_energy: float
    contradiction_ema: float
    plasticity: float
    rigidity: float
    repair_count: int
    dream_count: int
    mode_change_count: int
    narrative_tail: List[str]


class NarrativeSummaryResponse(BaseModel):
    text: str
    current_mode: str
    pressure: float
    salient_axes: List[str]


class StructuredMemoryContext(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    mode: str
    narrative_pressure: float
    intuition: List[str] = Field(default_factory=list)
    identity: IdentitySnapshotResponse
    narrative_summary: NarrativeSummaryResponse
    retrieval_hits: List[str] = Field(default_factory=list)
    overlay_hits: List[str] = Field(default_factory=list)
    evidence_refs: List[EvidenceRef] = Field(default_factory=list)


class RetrievalTraceSummary(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    query: str
    attractor_path_len: int
    hit_count: int
    overlay_hit_count: int
    ranked_kinds: List[str] = Field(default_factory=list)
    query_type: Optional[str] = None
    time_relation: Optional[str] = None
    time_start: Optional[float] = None
    time_end: Optional[float] = None
    time_anchor_event: Optional[str] = None


class ChatTimings(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    retrieval_ms: float
    generation_ms: float
    persist_ms: float
    total_ms: float


class ChatReplyAcceptedResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    reply: str
    event_id: str
    memory_context: StructuredMemoryContext
    rendered_memory_brief: str
    system_prompt: str
    user_prompt: str
    retrieval_trace_summary: RetrievalTraceSummary
    persistence: PersistenceReceiptResponse
    timings: ChatTimings
    llm_error: Optional[str] = None


class IdentityResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    identity: IdentitySnapshotResponse
    narrative_summary: NarrativeSummaryResponse


class EventStatusResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    event_id: str
    event_type: str
    session_id: str
    accepted_at: float
    projection: Optional[Dict[str, Any]] = None
    job: Optional[Dict[str, Any]] = None


class JobStatusResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    job_id: str
    job_type: str
    event_id: Optional[str] = None
    payload: Dict[str, Any]
    status: str
    attempts: int
    max_attempts: int
    available_at: float
    lease_owner: Optional[str] = None
    lease_expires_at: Optional[float] = None
    dedupe_key: Optional[str] = None
    created_ts: float
    updated_ts: float
    last_error: Optional[str] = None


class MemoryStatsResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    plot_count: int
    summary_count: int
    story_count: int
    theme_count: int
    architecture_mode: str
    current_mode: str
    pressure: float
    dream_count: int
    repair_count: int
    active_energy: float
    repressed_energy: float
    graph_metrics: Dict[str, Any] = Field(default_factory=dict)
    queue_depth: int
    oldest_pending_age_s: Optional[float] = None
    last_projected_seq: int
    last_fade_ts: Optional[float] = None
    last_evolve_ts: Optional[float] = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    timestamp: float
