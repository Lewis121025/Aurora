from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


SCHEMA_VERSION = "v4"


class MemoryKind(str, Enum):
    PLOT = "plot"
    STORY = "story"
    THEME = "theme"


class IngestRequest(BaseModel):
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


class RespondRequest(BaseModel):
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


class EvolveRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    dreams: Optional[int] = Field(default=None, ge=0, le=8)


class IngestResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    event_id: str
    plot_id: str
    story_id: Optional[str] = None
    mode: str
    source: Literal["wake", "dream", "repair", "mode"]
    tension: float
    contradiction: float
    active_energy: float
    repressed_energy: float


class QueryHit(BaseModel):
    id: str
    kind: MemoryKind
    score: float
    snippet: str
    metadata: Optional[Dict[str, str]] = None


class QueryResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    query: str
    attractor_path_len: int
    hits: List[QueryHit]


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
    evidence_refs: List[EvidenceRef] = Field(default_factory=list)


class RetrievalTraceSummary(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    query: str
    attractor_path_len: int
    hit_count: int
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
    ingest_ms: float
    total_ms: float


class ChatTurnResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    reply: str
    event_id: str
    memory_context: StructuredMemoryContext
    rendered_memory_brief: str
    system_prompt: str
    user_prompt: str
    retrieval_trace_summary: RetrievalTraceSummary
    ingest_result: IngestResponse
    timings: ChatTimings
    llm_error: Optional[str] = None


class IdentityResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    identity: IdentitySnapshotResponse
    narrative_summary: NarrativeSummaryResponse


class MemoryStatsResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    plot_count: int
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
    background_evolver: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    timestamp: float
