from __future__ import annotations

import numpy as np

from tests.conftest import FieldFactory


def test_replay_trains_predictor_and_promotes_procedure(field_factory: FieldFactory) -> None:
    field = field_factory()
    sequence = ["left", "right"] * 10
    for index, payload in enumerate(sequence, start=1):
        field.inject(
            {
                "payload": payload,
                "session_id": "session-a",
                "turn_id": f"turn-{index}",
                "source": "user",
            }
        )

    frames = [frame for frame in field.frames if frame.next_x is not None]
    assert len(frames) >= 10
    pre = float(np.mean([field.predictor.score_transition(frame) for frame in frames[:10]]))

    for _ in range(10):
        stats = field.maintenance_cycle(ms_budget=12)
        assert stats.replay_batch > 0

    post = float(np.mean([field.predictor.score_transition(frame) for frame in frames[:10]]))
    assert post < pre

    max_stability = max(trace.stability for trace in field.traces.values())
    assert max_stability > 0.5

    procedure_scores = [trace.role_logits.get("procedure", 0.0) for trace in field.traces.values()]
    assert max(procedure_scores) > 1.0

    option_edges = [edge for edge in field.edges.values() if edge.kind == "option"]
    assert option_edges
