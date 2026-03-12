from __future__ import annotations

from datetime import datetime, timezone

from aurora.core_math.contracts import FeedbackEnvelope, InputEnvelope, WakeEnvelope
from aurora.core_math.state import unseal_state
from aurora.substrate_core.engine import AuroraSubstrateCore


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
        WakeEnvelope(timestamp=now.replace(hour=12).isoformat()),
    )
    assert result.next_wake_at is not None
    assert result.health.substrate_alive is True


def test_feedback_returns_sealed_bytes() -> None:
    core = AuroraSubstrateCore()
    now = datetime(2026, 3, 11, 9, 0, tzinfo=timezone.utc)
    sealed = core.boot(now=now)
    input_result = core.on_input(
        sealed,
        InputEnvelope(user_text="tell me something", timestamp=now.isoformat(), language="en"),
    )
    new_sealed = core.on_feedback(
        input_result.sealed_state,
        FeedbackEnvelope(
            event_id=input_result.event_id,
            output_text="I said something back",
            timestamp=now.isoformat(),
            is_internal=False,
        ),
    )
    assert isinstance(new_sealed, bytes)
    state = unseal_state(new_sealed)
    # The spoken text must have been reincarnated as a spark
    spoken = [s for s in state.sparks if "[Spoke]" in s.text]
    assert len(spoken) >= 1


def test_health_snapshot_uses_truthful_substrate_field() -> None:
    core = AuroraSubstrateCore()
    now = datetime(2026, 3, 11, 9, 0, tzinfo=timezone.utc)
    sealed = core.boot(now=now)
    health = core.health_snapshot(sealed, last_error=None, provider_healthy=True)
    assert health.substrate_alive is True
