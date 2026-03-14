from __future__ import annotations

from pydantic import BaseModel

from aurora.phases.phase_types import Phase


class TurnRequest(BaseModel):
    session_id: str
    text: str


class TurnResponse(BaseModel):
    turn_id: str
    response_text: str
    touch_modes: tuple[str, ...]


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
    updated_at: float
    self_view: float
    world_view: float
    openness: float
    turns: int
    memory_fragments: int
    memory_traces: int
    memory_associations: int
    avg_salience: float
    avg_narrative_weight: float
    narrative_pressure: float
    sleep_cycles: int
    last_reweave_delta: float
    relation_moments: int
    relation_tone: str
    relation_strength: float
    transitions: int
