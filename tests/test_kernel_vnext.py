from __future__ import annotations

from aurora.runtime import AuroraField

from tests.conftest import FieldFactory


def test_inject_creates_packets_anchors_and_session_traces(field_factory: FieldFactory) -> None:
    field = field_factory()

    result = field.inject(
        {
            "session_id": "session-a",
            "turn_id": "turn-1",
            "source": "user",
            "payload_type": "text",
            "payload": "I live in Hangzhou. I like tea.",
            "meta": {"role": "user"},
        }
    )

    assert isinstance(field, AuroraField)
    assert result.packet_ids
    assert result.anchor_ids
    assert result.trace_ids
    assert len(field.anchor_store.packets) == len(result.packet_ids)
    assert len(field.anchor_store.anchors) == len(result.anchor_ids)
    assert len(field.trace_store.traces) == len(result.trace_ids)

    trace = field.trace_store.get(result.trace_ids[0])
    assert trace is not None
    assert trace.metadata["session_id"] == "session-a"
    assert trace.metadata["source"] == "user"
    assert list(trace.anchor_reservoir)


def test_maintenance_cycle_updates_replay_and_budget(field_factory: FieldFactory) -> None:
    field = field_factory()

    for turn_id, payload, source in (
        ("turn-1", "I live in Hangzhou.", "user"),
        ("turn-2", "I will remind you tomorrow.", "assistant"),
        ("turn-3", "I build Aurora systems.", "user"),
        ("turn-4", "Please remind me tomorrow.", "user"),
    ):
        field.inject(
            {
                "session_id": "session-a",
                "turn_id": turn_id,
                "source": source,
                "payload": payload,
                "meta": {"role": source},
            }
        )

    stats = field.maintenance_cycle(ms_budget=12)

    assert stats.elapsed_ms >= 0
    assert stats.replay_batch >= 0
    assert field.hot_trace_ids
    assert field.field_stats()["budget_state"]["max_traces"] == field.budget_config.max_traces


def test_snapshot_round_trip_restores_v5_kernel_state(field_factory: FieldFactory) -> None:
    field = field_factory()
    field.inject(
        {
            "session_id": "session-a",
            "turn_id": "turn-1",
            "source": "user",
            "payload": "I live in Hangzhou.",
            "meta": {"role": "user"},
        }
    )
    field.inject(
        {
            "session_id": "session-a",
            "turn_id": "turn-2",
            "source": "assistant",
            "payload": "I will remind you tomorrow.",
            "meta": {"role": "assistant"},
        }
    )
    field.maintenance_cycle(ms_budget=12)

    payload = field.to_snapshot_payload()
    restored = AuroraField.from_snapshot_payload(payload)

    assert payload["schema_version"] == 5
    assert restored.step == field.step
    assert len(restored.anchors) == len(field.anchors)
    assert len(restored.traces) == len(field.traces)
    assert len(restored.frames) == len(field.frames)
    assert restored.predictor.export_state().theta


def test_field_exposes_no_response_or_decoder_surface(field_factory: FieldFactory) -> None:
    field = field_factory()

    assert not hasattr(field, "respond")
    assert not hasattr(field, "local_decoder")
    assert not hasattr(field, "workspace_serializer")
