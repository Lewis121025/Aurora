from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4
import numpy as np

from aurora.core_math.contracts import (
    CollapseRequest, FeedbackEnvelope, HealthEnvelope,
    InputEnvelope, IntegrityEnvelope, ReleasedTrace, SubstrateInputResult,
    SubstrateWakeResult, WakeEnvelope,
)
from aurora.core_math.dynamics import (
    advance_latent_ou, boundary_budget, prediction_error,
    update_metric, verbosity_budget,
)
from aurora.core_math.encoding import HashingEncoder
from aurora.core_math.state import (
    ArrivalState, LatentState, MetricState, SEALED_STATE_VERSION,
    SealedState, SealedStateHeader, Spark,
    isoformat_utc, normalize, parse_utc, seal_state, unseal_state, utc_now,
)
from aurora.core_math.wake import (
    advance_arrival, observe_internal_action,
    observe_user_contact, sample_next_wake,
)


@dataclass(frozen=True)
class AuroraSubstrateConfig:
    latent_dim: int = 64
    metric_rank: int = 8
    rng_seed: int = 7
    sample_count: int = 3
    capacity: int = 1024  # 绝对物理极限：整个宇宙永远只有这 N 个粒子槽


class AuroraSubstrateCore:
    def __init__(self, config: AuroraSubstrateConfig | None = None) -> None:
        self.config = config or AuroraSubstrateConfig()
        self.encoder = HashingEncoder(dim=self.config.latent_dim, seed=self.config.rng_seed)

    def boot(self, now: datetime | None = None) -> bytes:
        start = now or utc_now()
        rng = np.random.default_rng(self.config.rng_seed)
        # 宇宙诞生时，所有槽位填充为温度为 0 的虚无星火
        sparks = [
            Spark(isoformat_utc(start), "", np.zeros(self.config.latent_dim), 0.0, "void")
            for _ in range(self.config.capacity)
        ]
        state = SealedState(
            header=SealedStateHeader(SEALED_STATE_VERSION, isoformat_utc(start), isoformat_utc(start)),
            latent=LatentState(normalize(rng.normal(scale=0.05, size=self.config.latent_dim))),
            metric=MetricState.isotropic(self.config.latent_dim, self.config.metric_rank),
            sparks=sparks,
            arrival=ArrivalState(isoformat_utc(start), 0.0, 0.2, 0.08, 0.05),
            rng_state=rng.bit_generator.state,
            last_event_time=isoformat_utc(start),
            next_wake_at=None,
        )
        state.next_wake_at = sample_next_wake(state.arrival, start, rng)
        return seal_state(state)

    def _thermodynamics(self, state: SealedState, dt_hours: float) -> np.ndarray:
        """核心物理引擎：熵增冷却 与 空间共振加热"""
        # 定律 1：熵增冷却——万物随时间失温
        decay = float(np.exp(-0.02 * dt_hours))
        for spark in state.sparks:
            spark.energy *= decay

        # 定律 2：共振加热——被扭曲空间照亮的粒子，吸收能量
        warped_latent = state.metric.matrix() @ state.latent.vector
        vectors = np.stack([spark.vector for spark in state.sparks])
        resonances = vectors @ warped_latent

        for i, spark in enumerate(state.sparks):
            if spark.text and resonances[i] > 0.05:
                spark.energy += 0.2 * float(resonances[i])

        return resonances

    def _reincarnate(
        self, state: SealedState, timestamp: str,
        text: str, vector: np.ndarray, error: float, source: str,
    ) -> None:
        """定律 3：夺舍新生——冲击力越大，初始温度越高，极难被淘汰"""
        coldest_idx = min(range(self.config.capacity), key=lambda i: state.sparks[i].energy)
        state.sparks[coldest_idx] = Spark(
            timestamp=timestamp,
            text=text,
            vector=vector,
            energy=1.0 + error * 5.0,
            source=source,
        )

    def on_input(self, sealed_state: bytes, envelope: InputEnvelope) -> SubstrateInputResult:
        state, rng = self._load(sealed_state)
        when = parse_utc(envelope.timestamp)
        dt = max(0.0, (when - parse_utc(state.last_event_time)).total_seconds() / 3600.0)

        state.latent.vector = advance_latent_ou(state.latent.vector, state.metric, dt, rng)
        advance_arrival(state.arrival, when)

        cue = self.encoder.encode(envelope.user_text)
        error = prediction_error(cue, state.latent.vector)

        # 运行热力学法则：冷却 + 共振
        resonances = self._thermodynamics(state, dt)

        # 意识上浮：共振最强的非空星火即是当前激活的记忆
        valid_idx = [i for i, spark in enumerate(state.sparks) if spark.text]
        if valid_idx:
            best_idx = sorted(valid_idx, key=lambda i: resonances[i], reverse=True)[:self.config.sample_count]
            resonated = [state.sparks[i] for i in best_idx]
        else:
            resonated = []

        # 定律 4：空间折叠——惊奇程度决定性格偏见的深度
        state.metric = update_metric(state.metric, cue, error)
        state.latent.vector = normalize(0.8 * state.latent.vector + 0.2 * cue)

        # 定律 3：夺舍
        self._reincarnate(state, envelope.timestamp, envelope.user_text, cue, error, "user")

        boundary = boundary_budget(error, sum(s.energy for s in resonated) * 0.1)
        verbosity = verbosity_budget(error, len(resonated))

        observe_user_contact(state.arrival, error)
        next_wake = sample_next_wake(state.arrival, when, rng)
        state.last_event_time = state.header.updated_at = isoformat_utc(when)
        state.next_wake_at = next_wake

        return SubstrateInputResult(
            sealed_state=self._save(state, rng),
            collapse_request=CollapseRequest(
                user_text=envelope.user_text,
                released_traces=[ReleasedTrace(text=s.text, source=s.source) for s in resonated],
                language=envelope.language,
                emit_reply=boundary < 0.96,
                boundary_budget=boundary,
                verbosity_budget=verbosity,
            ),
            event_id=uuid4().hex,
            next_wake_at=next_wake,
            health=self._health(state),
        )

    def on_feedback(self, sealed_state: bytes, envelope: FeedbackEnvelope) -> bytes:
        """行为反噬：无论说话还是做梦，都作为星火夺舍一具尸体"""
        state, rng = self._load(sealed_state)
        when = parse_utc(envelope.timestamp)

        if envelope.output_text:
            text = (
                f"[Dreamt]: {envelope.output_text}"
                if envelope.is_internal
                else f"[Spoke]: {envelope.output_text}"
            )
            cue = self.encoder.encode(text)
            state.metric = update_metric(state.metric, cue, error=0.05)
            state.latent.vector = normalize(0.9 * state.latent.vector + 0.1 * cue)
            self._reincarnate(
                state, envelope.timestamp, text, cue,
                error=0.0, source="dream" if envelope.is_internal else "self",
            )

        if envelope.is_internal:
            state.arrival.internal_drive = max(0.0, state.arrival.internal_drive - 0.4)
        elif not envelope.output_text:
            state.arrival.internal_drive = min(8.0, state.arrival.internal_drive + 0.1)

        state.last_event_time = state.header.updated_at = isoformat_utc(when)
        return self._save(state, rng)

    def on_wake(self, sealed_state: bytes, envelope: WakeEnvelope) -> SubstrateWakeResult:
        state, rng = self._load(sealed_state)
        when = parse_utc(envelope.timestamp)
        dt = max(0.0, (when - parse_utc(state.last_event_time)).total_seconds() / 3600.0)

        state.latent.vector = advance_latent_ou(state.latent.vector, state.metric, dt, rng)
        advance_arrival(state.arrival, when)

        resonances = self._thermodynamics(state, dt)
        valid_idx = [i for i, spark in enumerate(state.sparks) if spark.text]
        if valid_idx:
            best_idx = sorted(valid_idx, key=lambda i: resonances[i], reverse=True)[:self.config.sample_count]
            resonated = [state.sparks[i] for i in best_idx]
        else:
            resonated = []

        dream_req = None
        event_id = uuid4().hex

        if state.arrival.internal_drive > 0.7 and resonated:
            dream_req = CollapseRequest(
                user_text="<internal_dream>",
                released_traces=[ReleasedTrace(text=s.text, source=s.source) for s in resonated],
                language="auto",
                emit_reply=True,
                boundary_budget=1.0,
                verbosity_budget=1.0,
                is_internal_dream=True,
            )
            observe_internal_action(state.arrival, mass=0.5)

        next_wake = sample_next_wake(state.arrival, when, rng)
        state.last_event_time = state.header.updated_at = isoformat_utc(when)
        state.next_wake_at = next_wake

        return SubstrateWakeResult(
            sealed_state=self._save(state, rng),
            next_wake_at=next_wake,
            health=self._health(state),
            dream_request=dream_req,
            event_id=event_id,
        )

    def _load(self, blob: bytes) -> tuple[SealedState, np.random.Generator]:
        s = unseal_state(blob)
        r = np.random.default_rng()
        r.bit_generator.state = s.rng_state
        return s, r

    def _save(self, state: SealedState, rng: np.random.Generator) -> bytes:
        state.rng_state = rng.bit_generator.state
        return seal_state(state)

    def _health(self, s: SealedState, err: str | None = None, ph: bool = True) -> HealthEnvelope:
        alive_count = sum(1 for spark in s.sparks if spark.text)
        return HealthEnvelope(SEALED_STATE_VERSION, True, s.header.version, alive_count, s.next_wake_at, err, ph)

    def health_snapshot(self, sealed_state: bytes, last_error: str | None, provider_healthy: bool) -> HealthEnvelope:
        s, _ = self._load(sealed_state)
        return self._health(s, last_error, provider_healthy)

    def integrity_report(self, sealed_state: bytes, config_fingerprint: str) -> IntegrityEnvelope:
        s, _ = self._load(sealed_state)
        return build_integrity_report(
            sealed_state_version=s.header.version,
            config_fingerprint=config_fingerprint,
        )


def build_integrity_report(
    *,
    sealed_state_version: str = SEALED_STATE_VERSION,
    config_fingerprint: str,
    runtime_boundary: str = "process-opaque",
    substrate_transport: str = "in-process",
) -> IntegrityEnvelope:
    return IntegrityEnvelope(
        version=SEALED_STATE_VERSION,
        runtime_boundary=runtime_boundary,
        substrate_transport=substrate_transport,
        sealed_state_version=sealed_state_version,
        config_fingerprint=config_fingerprint,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
