from __future__ import annotations

from pydantic import BaseModel

from aurora.runtime.models import Phase


class TurnRequest(BaseModel):
    session_id: str
    text: str


class TurnResponse(BaseModel):
    turn_id: str
    response_text: str
    touch_channels: tuple[str, ...]


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
    sleep_need: float
    current_relation_id: str | None
    active_thread_ids: tuple[str, ...]
    active_knot_ids: tuple[str, ...]
    last_transition_at: float
    turns: int
    memory_fragments: int
    memory_traces: int
    memory_associations: int
    threads: int
    knots: int
    relation_moments: int
    trust: float
    boundary_tension: float
    sleep_cycles: int
    last_reweave_delta: float
    transitions: int
