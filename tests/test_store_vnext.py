from __future__ import annotations

import json
from pathlib import Path

from aurora.runtime import AuroraField

from tests.conftest import FieldFactory

def _json_default(value: object) -> object:
    if hasattr(value, "tolist"):
        return getattr(value, "tolist")()
    if hasattr(value, "__dict__"):
        return value.__dict__
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _inject_sample_field(field: AuroraField) -> None:
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


def test_snapshot_round_trips_schema_v5(tmp_path: Path, field_factory: FieldFactory) -> None:
    field = field_factory()
    _inject_sample_field(field)

    snapshot_path = tmp_path / "aurora-field.json"
    payload = field.to_snapshot_payload()
    snapshot_path.write_text(json.dumps(payload, default=_json_default, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    restored = AuroraField.from_snapshot_payload(payload)

    assert payload["schema_version"] == 5
    assert set(payload) == {
        "schema_version",
        "step",
        "current_ts",
        "config",
        "replay_config",
        "budget_config",
        "predictor_config",
        "packet_records",
        "anchors",
        "traces",
        "edges",
        "posterior_groups",
        "workspace_state",
        "frontier_state",
        "experience_frames",
        "predictor_state",
    }
    assert isinstance(payload["experience_frames"], list)
    assert payload["predictor_state"]["theta"]
    assert restored.step == field.step
    assert len(restored.anchor_store.packets) == len(field.anchor_store.packets)
    assert len(restored.anchor_store.anchors) == len(field.anchor_store.anchors)
    assert len(restored.trace_store.traces) == len(field.trace_store.traces)
    assert restored.field_stats()["trace_count"] == field.field_stats()["trace_count"]


def test_snapshot_payload_preserves_budget_and_replay_state(field_factory: FieldFactory) -> None:
    field = field_factory()
    _inject_sample_field(field)

    payload = field.to_snapshot_payload()

    assert payload["schema_version"] == 5
    assert payload["budget_config"]["max_traces"] == field.budget_config.max_traces
    assert isinstance(payload["experience_frames"], list)
    assert isinstance(payload["frontier_state"]["global"], dict)
