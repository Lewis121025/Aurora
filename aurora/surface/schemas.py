from __future__ import annotations

from pydantic import BaseModel

from aurora.runtime.contracts import Phase


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
    sleep_need: float
    active_relation_ids: tuple[str, ...]
    pending_sleep_relation_ids: tuple[str, ...]
    active_knot_ids: tuple[str, ...]
    anchor_thread_ids: tuple[str, ...]
    turns: int
    memory_fragments: int
    memory_traces: int
    memory_associations: int
    memory_threads: int
    memory_knots: int
    relation_formations: int
    relation_moments: int
    sleep_cycles: int
    transitions: int
