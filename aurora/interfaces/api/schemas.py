"""
Aurora API 模式
========================

Aurora V2 的唯一 API 契约。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


SCHEMA_VERSION = "v2"


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
    kinds: Optional[List[MemoryKind]] = None
    include_metadata: bool = False


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


class CausalChainRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    node_id: str
    direction: Literal["ancestors", "descendants"] = "ancestors"


class IngestResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    event_id: str
    plot_id: str
    story_id: Optional[str] = None
    memory_layer: Literal["explicit", "shadow"]
    tension: float
    surprise: float
    pred_error: float
    redundancy: float


class QueryHit(BaseModel):
    id: str
    kind: MemoryKind
    score: float
    snippet: str
    metadata: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    query: str
    attractor_path_len: int
    hits: List[QueryHit]
    latency_ms: Optional[float] = None


class EvidenceRef(BaseModel):
    id: str
    kind: MemoryKind
    score: float
    role: str


class StructuredMemoryContext(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    known_facts: List[str] = Field(default_factory=list)
    preferences: List[str] = Field(default_factory=list)
    relationship_state: List[str] = Field(default_factory=list)
    active_narratives: List[str] = Field(default_factory=list)
    temporal_context: List[str] = Field(default_factory=list)
    system_intuition: List[str] = Field(default_factory=list)
    cautions: List[str] = Field(default_factory=list)
    evidence_refs: List[EvidenceRef] = Field(default_factory=list)


class RetrievalTraceSummary(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

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


class CoherenceResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    overall_score: float = Field(..., ge=0.0, le=1.0)
    conflict_count: int = Field(..., ge=0)
    unfinished_story_count: int = Field(..., ge=0)
    recommendations: List[str]


class TraitBeliefResponse(BaseModel):
    probability: float
    description: str


class RelationshipBeliefResponse(BaseModel):
    trust: float
    familiarity: float
    interaction_count: int


class SubconsciousResponse(BaseModel):
    dark_matter_count: int
    repressed_count: int
    last_intuition: List[str] = Field(default_factory=list)


class SelfNarrativeResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    profile_id: str
    identity_statement: str
    identity_narrative: str
    seed_narrative: str
    capability_narrative: str
    core_values: List[str]
    coherence_score: float = Field(..., ge=0.0, le=1.0)
    capabilities: Dict[str, Dict[str, Any]]
    trait_beliefs: Dict[str, TraitBeliefResponse]
    relationships: Dict[str, RelationshipBeliefResponse]
    subconscious: SubconsciousResponse
    unresolved_tensions: List[str]
    full_narrative: str


class CausalChainItem(BaseModel):
    node_id: str
    strength: float = Field(..., ge=0.0, le=1.0)


class CausalChainResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    chain: List[CausalChainItem]


class MemoryStatsResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    plot_count: int = Field(..., ge=0)
    story_count: int = Field(..., ge=0)
    theme_count: int = Field(..., ge=0)
    coherence_score: float = Field(..., ge=0.0, le=1.0)
    self_narrative_coherence: float = Field(..., ge=0.0, le=1.0)
    gate_pass_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    cluster_entropy: Optional[float] = Field(default=None, ge=0.0)


class ErrorResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    error: str
    message: str
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    timestamp: float
    components: Optional[Dict[str, str]] = None


def get_schema_version() -> str:
    return SCHEMA_VERSION


def list_schema_versions() -> List[str]:
    return [SCHEMA_VERSION]
