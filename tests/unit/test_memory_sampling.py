from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from aurora.core_math.encoding import HashingEncoder
from aurora.core_math.memory import MemoryField
from aurora.core_math.state import (
    ArrivalState,
    LatentState,
    MetricState,
    SEALED_STATE_VERSION,
    SealedState,
    SealedStateHeader,
    isoformat_utc,
)


def _blank_state() -> SealedState:
    now = datetime(2026, 3, 11, 9, 0, tzinfo=timezone.utc)
    return SealedState(
        header=SealedStateHeader(
            version=SEALED_STATE_VERSION,
            created_at=isoformat_utc(now),
            updated_at=isoformat_utc(now),
        ),
        latent=LatentState(vector=np.zeros(16, dtype=np.float64)),
        metric=MetricState.isotropic(dim=16, rank=4),
        memory={},
        recent_fiber_ids=[],
        arrival=ArrivalState(
            last_event_time=isoformat_utc(now),
            no_contact_hours=0.0,
            internal_drive=0.0,
            decay_per_hour=0.1,
            base_rate=0.05,
        ),
        rng_state=np.random.default_rng(1).bit_generator.state,
        last_event_time=isoformat_utc(now),
    )


def test_sampling_is_distributional_not_ranked() -> None:
    encoder = HashingEncoder(dim=16, seed=3)
    memory = MemoryField(encoder)
    state = _blank_state()
    now = datetime(2026, 3, 11, 9, 0, tzinfo=timezone.utc)
    memory.add_dialogue_trace(state, now, "alpha memory", "alpha memory", encoder.encode("alpha"))
    memory.add_dialogue_trace(state, now, "beta memory", "beta memory", encoder.encode("beta"))
    cue = encoder.encode("memory")
    rng = np.random.default_rng(9)

    seen: set[str] = set()
    for _ in range(50):
        sampled = memory.sample(state, cue, np.zeros(16), rng, count=1, steps=5)
        assert sampled
        seen.add(sampled[0].fiber_id)
    assert len(seen) == 2


def test_virtual_trace_does_not_modify_anchor() -> None:
    encoder = HashingEncoder(dim=16, seed=3)
    memory = MemoryField(encoder)
    state = _blank_state()
    now = datetime(2026, 3, 11, 9, 0, tzinfo=timezone.utc)
    anchor_id = memory.add_dialogue_trace(
        state,
        now,
        "this anchor should remain stable",
        "this anchor should remain stable",
        encoder.encode("stable"),
    )
    original = state.memory[anchor_id].anchor.raw_text
    memory.reconsolidate(state, [anchor_id], now, "new cue")
    virtual = memory.virtual_trace_text(state.memory[anchor_id])
    assert state.memory[anchor_id].anchor.raw_text == original
    assert "schema echo" in virtual
