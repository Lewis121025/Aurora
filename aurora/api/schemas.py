"""
Aurora API Schemas (Versioned)
==============================

Provides versioned Pydantic models for API contracts.

Version History:
- v1: Initial stable API (current)

Usage:
    from aurora.api.schemas import IngestRequestV1, QueryResponseV1
    
    request = IngestRequestV1(
        event_id="evt_123",
        user_id="user_456",
        ...
    )

Schema versioning allows:
- Backward compatible changes within a version
- Multiple versions to coexist during migration
- Clear API contracts between frontend/backend
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Enums
# =============================================================================

class MemoryKind(str, Enum):
    """Types of memory units."""
    PLOT = "plot"
    STORY = "story"
    THEME = "theme"


class PlotStatus(str, Enum):
    """Plot lifecycle status."""
    ACTIVE = "active"
    ABSORBED = "absorbed"
    ARCHIVED = "archived"


class StoryStatus(str, Enum):
    """Story lifecycle status."""
    DEVELOPING = "developing"
    RESOLVED = "resolved"
    ABANDONED = "abandoned"


class ThemeType(str, Enum):
    """Types of themes."""
    PATTERN = "pattern"
    LESSON = "lesson"
    PREFERENCE = "preference"
    CAUSALITY = "causality"
    CAPABILITY = "capability"
    LIMITATION = "limitation"


# =============================================================================
# V1 Request Models
# =============================================================================

class IngestRequestV1(BaseModel):
    """V1 Ingest Request - Store a new interaction in memory."""
    
    event_id: Optional[str] = Field(
        default=None,
        description="Unique event identifier. Auto-generated if not provided.",
        examples=["evt_abc123"],
    )
    user_id: str = Field(
        ...,
        description="User identifier for multi-tenant isolation.",
        examples=["user_456"],
    )
    session_id: str = Field(
        ...,
        description="Session identifier for grouping related interactions.",
        examples=["sess_789"],
    )
    user_message: str = Field(
        ...,
        description="The user's message or input.",
        min_length=1,
        examples=["How do I implement a memory system?"],
    )
    agent_message: str = Field(
        ...,
        description="The agent's response or action taken.",
        min_length=1,
        examples=["You can use a narrative memory architecture..."],
    )
    actors: Optional[List[str]] = Field(
        default=None,
        description="List of participants in the interaction.",
        examples=[["user", "agent"]],
    )
    context: Optional[str] = Field(
        default=None,
        description="Additional context about the interaction.",
        examples=["programming discussion"],
    )
    ts: Optional[float] = Field(
        default=None,
        description="Unix timestamp of the interaction. Auto-set if not provided.",
    )
    
    class Config:
        json_schema_extra = {
            "version": "v1",
            "example": {
                "event_id": "evt_abc123",
                "user_id": "user_456",
                "session_id": "sess_789",
                "user_message": "How do I implement a memory system?",
                "agent_message": "You can use a narrative memory architecture...",
                "actors": ["user", "agent"],
                "context": "programming discussion",
            }
        }


class QueryRequestV1(BaseModel):
    """V1 Query Request - Search memories by semantic similarity."""
    
    user_id: str = Field(
        ...,
        description="User identifier for multi-tenant isolation.",
    )
    text: str = Field(
        ...,
        description="Natural language search query.",
        min_length=1,
        examples=["How do I avoid hard-coded thresholds?"],
    )
    k: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Number of results to return.",
    )
    kinds: Optional[List[MemoryKind]] = Field(
        default=None,
        description="Filter by memory types. None means all types.",
        examples=[["plot", "story"]],
    )
    include_metadata: bool = Field(
        default=False,
        description="Include detailed metadata in response.",
    )
    
    class Config:
        json_schema_extra = {"version": "v1"}


class FeedbackRequestV1(BaseModel):
    """V1 Feedback Request - Provide feedback on retrieval quality."""
    
    user_id: str = Field(
        ...,
        description="User identifier.",
    )
    query_text: str = Field(
        ...,
        description="The original search query.",
    )
    chosen_id: str = Field(
        ...,
        description="ID of the memory that was chosen/useful.",
    )
    success: bool = Field(
        ...,
        description="Whether the retrieval was successful/helpful.",
    )
    
    class Config:
        json_schema_extra = {"version": "v1"}


class EvolveRequestV1(BaseModel):
    """V1 Evolve Request - Trigger memory evolution."""
    
    user_id: str = Field(
        ...,
        description="User identifier.",
    )
    
    class Config:
        json_schema_extra = {"version": "v1"}


class CausalChainRequestV1(BaseModel):
    """V1 Causal Chain Request - Get causal relationships."""
    
    user_id: str = Field(
        ...,
        description="User identifier.",
    )
    node_id: str = Field(
        ...,
        description="Node ID to trace causal chain from.",
    )
    direction: Literal["ancestors", "descendants"] = Field(
        default="ancestors",
        description="Direction to trace.",
    )
    
    class Config:
        json_schema_extra = {"version": "v1"}


# =============================================================================
# V1 Response Models
# =============================================================================

class IngestResponseV1(BaseModel):
    """V1 Ingest Response - Result of storing an interaction."""
    
    event_id: str = Field(
        ...,
        description="Event identifier (input or generated).",
    )
    plot_id: str = Field(
        ...,
        description="ID of the created plot.",
    )
    story_id: Optional[str] = Field(
        default=None,
        description="ID of the assigned story, if encoded.",
    )
    encoded: bool = Field(
        ...,
        description="Whether the plot was stored (gate decision).",
    )
    tension: float = Field(
        ...,
        description="Narrative tension score.",
    )
    surprise: float = Field(
        ...,
        description="Surprise score (information novelty).",
    )
    pred_error: float = Field(
        ...,
        description="Prediction error vs story model.",
    )
    redundancy: float = Field(
        ...,
        description="Redundancy with existing plots.",
    )
    
    class Config:
        json_schema_extra = {"version": "v1"}


class QueryHitV1(BaseModel):
    """V1 Query Hit - A single search result."""
    
    id: str = Field(
        ...,
        description="Memory unit ID.",
    )
    kind: MemoryKind = Field(
        ...,
        description="Type of memory unit.",
    )
    score: float = Field(
        ...,
        description="Relevance score.",
    )
    snippet: str = Field(
        ...,
        description="Text snippet or summary.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata if requested.",
    )


class QueryResponseV1(BaseModel):
    """V1 Query Response - Search results."""
    
    query: str = Field(
        ...,
        description="The original search query.",
    )
    attractor_path_len: int = Field(
        ...,
        description="Length of attractor convergence path.",
    )
    hits: List[QueryHitV1] = Field(
        ...,
        description="List of matching memories.",
    )
    latency_ms: Optional[float] = Field(
        default=None,
        description="Query latency in milliseconds.",
    )
    
    class Config:
        json_schema_extra = {"version": "v1"}


class CoherenceResponseV1(BaseModel):
    """V1 Coherence Response - Memory coherence check results."""
    
    overall_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall coherence score (0-1).",
    )
    conflict_count: int = Field(
        ...,
        ge=0,
        description="Number of detected conflicts.",
    )
    unfinished_story_count: int = Field(
        ...,
        ge=0,
        description="Number of unresolved stories.",
    )
    recommendations: List[str] = Field(
        ...,
        description="Recommended actions to improve coherence.",
    )
    
    class Config:
        json_schema_extra = {"version": "v1"}


class SelfNarrativeResponseV1(BaseModel):
    """V1 Self Narrative Response - Agent's self-model."""
    
    identity_statement: str = Field(
        ...,
        description="Short identity statement.",
    )
    identity_narrative: str = Field(
        ...,
        description="Detailed identity narrative.",
    )
    capability_narrative: str = Field(
        ...,
        description="Description of capabilities.",
    )
    core_values: List[str] = Field(
        ...,
        description="List of core values.",
    )
    coherence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Narrative coherence score.",
    )
    capabilities: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="Capability beliefs with probabilities.",
    )
    relationships: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="Relationship beliefs.",
    )
    unresolved_tensions: List[str] = Field(
        ...,
        description="Current unresolved tensions.",
    )
    full_narrative: str = Field(
        ...,
        description="Complete narrative text.",
    )
    
    class Config:
        json_schema_extra = {"version": "v1"}


class CausalChainItemV1(BaseModel):
    """V1 Causal Chain Item - A node in the causal chain."""
    
    node_id: str
    strength: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Causal relationship strength.",
    )


class CausalChainResponseV1(BaseModel):
    """V1 Causal Chain Response - Causal relationship chain."""
    
    chain: List[CausalChainItemV1] = Field(
        ...,
        description="List of causally related nodes.",
    )
    
    class Config:
        json_schema_extra = {"version": "v1"}


class MemoryStatsResponseV1(BaseModel):
    """V1 Memory Stats Response - Memory system statistics."""
    
    plot_count: int = Field(
        ...,
        ge=0,
        description="Number of stored plots.",
    )
    story_count: int = Field(
        ...,
        ge=0,
        description="Number of stories.",
    )
    theme_count: int = Field(
        ...,
        ge=0,
        description="Number of themes.",
    )
    coherence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall coherence score.",
    )
    self_narrative_coherence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Self-narrative coherence.",
    )
    gate_pass_rate: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Memory gate pass rate.",
    )
    cluster_entropy: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Story cluster entropy.",
    )
    
    class Config:
        json_schema_extra = {"version": "v1"}


class ErrorResponseV1(BaseModel):
    """V1 Error Response - Standard error format."""
    
    error: str = Field(
        ...,
        description="Error type or code.",
    )
    message: str = Field(
        ...,
        description="Human-readable error message.",
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details.",
    )
    
    class Config:
        json_schema_extra = {"version": "v1"}


class HealthResponseV1(BaseModel):
    """V1 Health Response - Service health check."""
    
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ...,
        description="Service status.",
    )
    version: str = Field(
        ...,
        description="Service version.",
    )
    timestamp: float = Field(
        ...,
        description="Check timestamp.",
    )
    components: Optional[Dict[str, str]] = Field(
        default=None,
        description="Component-level status.",
    )
    
    class Config:
        json_schema_extra = {"version": "v1"}


# =============================================================================
# Type Aliases for Convenience
# =============================================================================

# Current version aliases (update these when releasing new versions)
IngestRequest = IngestRequestV1
IngestResponse = IngestResponseV1
QueryRequest = QueryRequestV1
QueryResponse = QueryResponseV1
QueryHit = QueryHitV1
FeedbackRequest = FeedbackRequestV1
EvolveRequest = EvolveRequestV1
CoherenceResponse = CoherenceResponseV1
SelfNarrativeResponse = SelfNarrativeResponseV1
CausalChainRequest = CausalChainRequestV1
CausalChainResponse = CausalChainResponseV1
MemoryStatsResponse = MemoryStatsResponseV1
ErrorResponse = ErrorResponseV1
HealthResponse = HealthResponseV1


# =============================================================================
# Schema Version Info
# =============================================================================

SCHEMA_VERSION = "v1"
SCHEMA_VERSIONS = ["v1"]

def get_schema_version() -> str:
    """Get current schema version."""
    return SCHEMA_VERSION

def list_schema_versions() -> List[str]:
    """List all available schema versions."""
    return SCHEMA_VERSIONS.copy()
