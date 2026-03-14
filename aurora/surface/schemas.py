from __future__ import annotations

from pydantic import BaseModel

from aurora.runtime.models import Phase


class TurnRequest(BaseModel):
    session_id: str
    text: str


class TurnResponse(BaseModel):
    turn_id: str
    response_text: str
    aurora_move: str
    dominant_channels: tuple[str, ...]


class PhaseResponse(BaseModel):
    phase: Phase
    transition_id: str


class HealthResponse(BaseModel):
    status: str
    phase: str
    turns: int
    transitions: int


class StateResponse(BaseModel):
    phase: str
    continuity_pressure: float
    sleep_pressure: float
    coherence_pressure: float
    softness: float
    boundary_tension: float
    active_relation_id: str | None
    recent_chapter_bias: tuple[str, ...]
    turns: int
    memory_fragments: int
    memory_traces: int
    memory_associations: int
    memory_chapters: int
    relation_count: int
    sleep_cycles: int
    last_reweave_delta: float
    transitions: int
