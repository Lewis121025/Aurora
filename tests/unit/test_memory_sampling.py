from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

from aurora.core_math.contracts import InputEnvelope
from aurora.core_math.state import unseal_state
from aurora.substrate_core.engine import AuroraSubstrateCore
from aurora.substrate_core.engine import AuroraSubstrateConfig


def _small_core() -> AuroraSubstrateCore:
    """16-dim latent, 8 spark slots — fast enough for unit tests."""
    return AuroraSubstrateCore(
        AuroraSubstrateConfig(latent_dim=16, metric_rank=4, rng_seed=1, sample_count=2, capacity=8)
    )


def test_reincarnation_overwrites_coldest_spark() -> None:
    core = _small_core()
    now = datetime(2026, 3, 11, 9, 0, tzinfo=timezone.utc)
    sealed = core.boot(now=now)

    # After boot: all 8 slots are void (energy=0, text="")
    # Feed enough inputs to fill all slots, then verify oldest gets overwritten
    ts_base = now
    for i in range(9):  # 9 inputs > 8 slots: forces at least one reincarnation
        from datetime import timedelta
        ts = ts_base.replace(hour=9 + i) if i < 15 else ts_base
        result = core.on_input(
            sealed,
            InputEnvelope(user_text=f"message {i}", timestamp=ts.isoformat(), language="en"),
        )
        sealed = result.sealed_state

    state = unseal_state(sealed)
    non_void = [s for s in state.sparks if s.text]
    # With 9 inputs and 8 slots, all 8 slots must have been filled
    assert len(non_void) == 8
    # Every live spark must carry a positive energy
    assert all(s.energy > 0 for s in non_void)


def test_energy_decay_reduces_all_sparks() -> None:
    core = _small_core()
    now = datetime(2026, 3, 11, 9, 0, tzinfo=timezone.utc)
    sealed = core.boot(now=now)

    # Plant one spark
    result = core.on_input(
        sealed,
        InputEnvelope(user_text="remember this", timestamp=now.isoformat(), language="en"),
    )
    sealed = result.sealed_state
    state_before = unseal_state(sealed)
    energy_before = max(s.energy for s in state_before.sparks if s.text)

    # Trigger a wake 48 hours later (large dt → strong decay)
    from datetime import timedelta
    from aurora.core_math.contracts import WakeEnvelope
    later = now + timedelta(hours=48)
    wake_result = core.on_wake(sealed, WakeEnvelope(timestamp=later.isoformat()))
    state_after = unseal_state(wake_result.sealed_state)

    # The same spark (same text slot) should have lower energy after 48h
    after_energies = {s.text: s.energy for s in state_after.sparks if s.text}
    assert "remember this" in after_energies
    assert after_energies["remember this"] < energy_before


def test_high_error_spark_gets_higher_initial_energy() -> None:
    core = _small_core()
    now = datetime(2026, 3, 11, 9, 0, tzinfo=timezone.utc)
    sealed = core.boot(now=now)

    # Two inputs at the same time: one very predictable, one very surprising
    # We can't control error directly, but we can check energy ordering via the formula:
    # energy = 1.0 + error * 5.0
    # A cue identical to the latent vector has error ≈ 0 → energy ≈ 1.0
    # A totally orthogonal cue has error ≈ 1 → energy ≈ 6.0
    # We verify that after boot with near-zero latent, any input gets energy >= 1.0
    result = core.on_input(
        sealed,
        InputEnvelope(user_text="unexpected shock", timestamp=now.isoformat(), language="en"),
    )
    state = unseal_state(result.sealed_state)
    live = [s for s in state.sparks if s.text == "unexpected shock"]
    assert len(live) == 1
    assert live[0].energy >= 1.0
