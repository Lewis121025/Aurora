from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

import numpy as np

from aurora.core_math.contracts import (
    CollapseRequest,
    HealthEnvelope,
    InputEnvelope,
    ReleasedTrace,
    SubstrateInputResult,
    SubstrateWakeResult,
    WakeEnvelope,
)
from aurora.core_math.dynamics import (
    advance_latent_ou,
    anisotropic_gradient,
    boundary_budget,
    prediction_error,
    update_metric,
    verbosity_budget,
)
from aurora.core_math.encoding import HashingEncoder
from aurora.core_math.memory import MemoryField
from aurora.core_math.sealing import seal_state, unseal_state
from aurora.core_math.state import (
    ArrivalState,
    LatentState,
    MetricState,
    SEALED_STATE_VERSION,
    SealedState,
    SealedStateHeader,
    isoformat_utc,
    parse_utc,
    utc_now,
)
from aurora.core_math.wake import (
    advance_arrival,
    no_contact_surprisal,
    observe_internal_action,
    observe_user_contact,
    sample_next_wake,
)


@dataclass(frozen=True)
class AuroraSubstrateConfig:
    latent_dim: int = 32
    metric_rank: int = 4
    rng_seed: int = 7
    sample_count: int = 3
    sample_steps: int = 12
    trace_limit: int = 6


class AuroraSubstrateCore:
    def __init__(self, config: AuroraSubstrateConfig | None = None) -> None:
        self.config = config or AuroraSubstrateConfig()
        self.encoder = HashingEncoder(dim=self.config.latent_dim, seed=self.config.rng_seed)
        self.memory = MemoryField(self.encoder, trace_limit=self.config.trace_limit)

    def boot(self, now: datetime | None = None) -> bytes:
        start = now or utc_now()
        rng = np.random.default_rng(self.config.rng_seed)
        state = SealedState(
            header=SealedStateHeader(
                version=SEALED_STATE_VERSION,
                created_at=isoformat_utc(start),
                updated_at=isoformat_utc(start),
            ),
            latent=LatentState(vector=rng.normal(scale=0.05, size=self.config.latent_dim)),
            metric=MetricState.isotropic(
                dim=self.config.latent_dim,
                rank=self.config.metric_rank,
            ),
            memory={},
            recent_fiber_ids=[],
            arrival=ArrivalState(
                last_event_time=isoformat_utc(start),
                no_contact_hours=0.0,
                internal_drive=0.2,
                decay_per_hour=0.08,
                base_rate=0.05,
            ),
            rng_state=rng.bit_generator.state,
            last_event_time=isoformat_utc(start),
            next_wake_at=None,
        )
        state.next_wake_at = sample_next_wake(state.arrival, start, rng)
        state.rng_state = rng.bit_generator.state
        return seal_state(state)

    def on_input(self, sealed_state: bytes, envelope: InputEnvelope) -> SubstrateInputResult:
        state, rng = self._load(sealed_state)
        when = parse_utc(envelope.timestamp)
        advance_arrival(state.arrival, when)
        delta = max(0.0, (when - parse_utc(state.last_event_time)).total_seconds() / 3600.0)
        state.latent.vector = advance_latent_ou(state.latent.vector, state.metric, delta, rng)

        cue = self.encoder.encode(envelope.user_text)
        sampled = self.memory.sample(
            state,
            cue_embedding=cue,
            latent_embedding=state.latent.vector,
            rng=rng,
            count=self.config.sample_count,
            steps=self.config.sample_steps,
        )
        sampled_embeddings = [
            state.memory[item.fiber_id].centroid
            for item in sampled
            if item.fiber_id in state.memory
        ]
        gradient = anisotropic_gradient(cue, state.latent.vector, sampled_embeddings)
        state.latent.vector = np.tanh(state.latent.vector + 0.9 * gradient)
        state.metric = update_metric(state.metric, gradient)

        error = prediction_error(cue, state.latent.vector)
        basin_pressure = self.memory.reinforce(
            state,
            fiber_ids=[item.fiber_id for item in sampled],
            error=error,
        )
        self.memory.reconsolidate(
            state,
            fiber_ids=[item.fiber_id for item in sampled],
            when=when,
            user_text=envelope.user_text,
        )
        released_traces = [
            ReleasedTrace(text=text, source="trace")
            for text in self.memory.released_traces(state, sampled)
        ]
        released_virtual = [
            ReleasedTrace(text=text, source="virtual")
            for text in self.memory.released_virtual_traces(sampled)
        ]
        boundary = boundary_budget(error, basin_pressure)
        verbosity = verbosity_budget(error, len(released_traces))
        emit_reply = boundary < 0.96

        released_context = " ".join(item.text for item in released_traces[:2]).strip()
        if not released_context:
            released_context = envelope.user_text
        self.memory.add_dialogue_trace(
            state,
            when=when,
            user_text=envelope.user_text,
            released_text=released_context,
            cue_embedding=cue,
        )

        observe_user_contact(state.arrival, error)
        next_wake_at = sample_next_wake(state.arrival, when, rng)
        state.last_event_time = isoformat_utc(when)
        state.header.updated_at = isoformat_utc(when)
        state.next_wake_at = next_wake_at
        state.rng_state = rng.bit_generator.state
        result_state = seal_state(state)
        health = self._health(state)
        return SubstrateInputResult(
            sealed_state=result_state,
            collapse_request=CollapseRequest(
                user_text=envelope.user_text,
                released_traces=released_traces,
                released_virtual_traces=released_virtual,
                language=envelope.language,
                emit_reply=emit_reply,
                boundary_budget=boundary,
                verbosity_budget=verbosity,
            ),
            event_id=uuid4().hex,
            next_wake_at=next_wake_at,
            health=health,
        )

    def on_wake(self, sealed_state: bytes, envelope: WakeEnvelope) -> SubstrateWakeResult:
        state, rng = self._load(sealed_state)
        when = parse_utc(envelope.timestamp)
        advance_arrival(state.arrival, when)
        delta = max(0.0, (when - parse_utc(state.last_event_time)).total_seconds() / 3600.0)
        state.latent.vector = advance_latent_ou(state.latent.vector, state.metric, delta, rng)

        cue = np.zeros(self.config.latent_dim, dtype=np.float64)
        cue[: min(4, cue.shape[0])] = 1.0
        sampled = self.memory.sample(
            state,
            cue_embedding=cue,
            latent_embedding=state.latent.vector,
            rng=rng,
            count=max(1, self.config.sample_count - 1),
            steps=self.config.sample_steps,
        )
        internal_note = " ".join(
            [item.virtual_trace or item.released_trace for item in sampled][:2]
        ).strip()
        if not internal_note:
            internal_note = f"silent replay after {no_contact_surprisal(state.arrival):.3f}"
        self.memory.add_internal_trace(state, when=when, narrative=internal_note)
        observe_internal_action(state.arrival, mass=max(0.25, len(sampled) * 0.15))

        next_wake_at = sample_next_wake(state.arrival, when, rng)
        state.last_event_time = isoformat_utc(when)
        state.header.updated_at = isoformat_utc(when)
        state.next_wake_at = next_wake_at
        state.rng_state = rng.bit_generator.state
        result_state = seal_state(state)
        return SubstrateWakeResult(
            sealed_state=result_state,
            next_wake_at=next_wake_at,
            health=self._health(state),
        )

    def health_snapshot(
        self,
        sealed_state: bytes,
        last_error: str | None,
        provider_healthy: bool,
    ) -> HealthEnvelope:
        state, _ = self._load(sealed_state)
        health = self._health(state)
        return HealthEnvelope(
            version=health.version,
            substrate_alive=health.substrate_alive,
            sealed_state_version=health.sealed_state_version,
            anchor_count=health.anchor_count,
            next_wake_at=health.next_wake_at,
            last_error=last_error,
            provider_healthy=provider_healthy,
        )

    def _load(self, sealed_state: bytes) -> tuple[SealedState, np.random.Generator]:
        state = unseal_state(sealed_state)
        rng = np.random.default_rng()
        rng.bit_generator.state = state.rng_state
        return state, rng

    def _health(self, state: SealedState) -> HealthEnvelope:
        return HealthEnvelope(
            version=SEALED_STATE_VERSION,
            substrate_alive=True,
            sealed_state_version=state.header.version,
            anchor_count=len(state.memory),
            next_wake_at=state.next_wake_at,
            last_error=None,
            provider_healthy=True,
        )
