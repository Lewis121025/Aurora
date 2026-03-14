from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ReleasedTrace:
    text: str
    source: str


@dataclass(frozen=True)
class InputEnvelope:
    user_text: str
    timestamp: str
    language: str = "auto"
    media_refs: list[str] | None = None
    kinesthetic_entropy: float = 0.5  # 0.0=synthetic/copy-paste, 1.0=erratic typing
    hour_of_day: float = 12.0  # 0.0-23.99


@dataclass(frozen=True)
class WakeEnvelope:
    timestamp: str


@dataclass(frozen=True)
class FeedbackEnvelope:
    event_id: str
    output_text: str | None
    timestamp: str
    is_internal: bool = False
    is_compression: bool = False
    consumed_nodes: list[str] | None = None


@dataclass(frozen=True)
class CollapseRequest:
    user_text: str
    released_traces: list[ReleasedTrace]
    language: str
    emit_reply: bool
    boundary_budget: float
    verbosity_budget: float
    is_internal_dream: bool = False
    is_internal_dream_compression: bool = False
    vad_state: list[float] | None = None  # [Valence, Arousal, Dominance] of current mood
    mutual_respect: float = 0.0
    media_refs: list[str] | None = None


@dataclass(frozen=True)
class CollapseResult:
    output_text: str | None
    provider_name: str


@dataclass(frozen=True)
class HealthEnvelope:
    version: str
    substrate_alive: bool
    sealed_state_version: str
    anchor_count: int
    next_wake_at: str | None
    last_error: str | None
    provider_healthy: bool


@dataclass(frozen=True)
class IntegrityEnvelope:
    version: str
    runtime_boundary: str
    substrate_transport: str
    sealed_state_version: str
    config_fingerprint: str
    generated_at: str


@dataclass(frozen=True)
class SubstrateInputResult:
    sealed_state: bytes
    collapse_request: CollapseRequest
    event_id: str
    next_wake_at: str | None
    health: HealthEnvelope


@dataclass(frozen=True)
class SubstrateWakeResult:
    sealed_state: bytes
    next_wake_at: str | None
    health: HealthEnvelope
    dream_request: CollapseRequest | None = None
    event_id: str | None = None
    consumed_nodes: list[str] | None = None


@dataclass(frozen=True)
class InputOutcome:
    event_id: str
    output_text: str | None
    outcome: str
    next_wake_at: str | None
