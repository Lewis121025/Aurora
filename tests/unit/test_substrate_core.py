from __future__ import annotations

from datetime import datetime, timezone

from aurora.core_math.contracts import InputEnvelope, WakeEnvelope
from aurora.substrate_core import AuroraSubstrateCore


def test_input_path_emits_released_context_without_internal_state() -> None:
    core = AuroraSubstrateCore()
    now = datetime(2026, 3, 11, 9, 0, tzinfo=timezone.utc)
    sealed = core.boot(now=now)
    result = core.on_input(
        sealed,
        InputEnvelope(user_text="hello aurora", timestamp=now.isoformat(), language="en"),
    )
    assert result.event_id
    assert result.health.anchor_count >= 1
    payload = repr(result.collapse_request).lower()
    assert "latent" not in payload
    assert "metric" not in payload


def test_wake_path_schedules_next_wake() -> None:
    core = AuroraSubstrateCore()
    now = datetime(2026, 3, 11, 9, 0, tzinfo=timezone.utc)
    sealed = core.boot(now=now)
    result = core.on_wake(
        sealed,
        WakeEnvelope(timestamp=(now.replace(hour=12)).isoformat()),
    )
    assert result.next_wake_at is not None
    assert result.health.substrate_alive is True


def test_health_snapshot_uses_truthful_substrate_field() -> None:
    core = AuroraSubstrateCore()
    now = datetime(2026, 3, 11, 9, 0, tzinfo=timezone.utc)
    sealed = core.boot(now=now)
    health = core.health_snapshot(sealed, last_error=None, provider_healthy=True)
    assert health.substrate_alive is True
