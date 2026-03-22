from __future__ import annotations

import numpy as np

from aurora.core.types import Anchor
from aurora.core.config import FieldConfig
from aurora.runtime import AuroraField

from tests.conftest import FieldFactory


def test_primary_actions_cover_birth_assimilate_split_and_birth_again(field_factory: FieldFactory) -> None:
    del field_factory
    field = AuroraField(
        FieldConfig(
            latent_dim=2,
            context_dim=2,
            candidate_size=8,
            workspace_size=4,
            cue_context_weight=1.0,
            frontier_context_weight=0.2,
            predictor_context_weight=0.0,
        )
    )

    first_anchor = Anchor(
        anchor_id="a1",
        packet_id="p1",
        session_id="session-a",
        turn_id="turn-1",
        source="user",
        z=np.array([1.0, 0.0]),
        ts=1.0,
    )
    first = field._score_primary_actions(first_anchor, pred_mu=np.zeros(2, dtype=np.float64))
    assert first.action == "BIRTH"
    trace_id = field._apply_primary_action(first, first_anchor)

    second_anchor = Anchor(
        anchor_id="a2",
        packet_id="p2",
        session_id="session-a",
        turn_id="turn-2",
        source="user",
        z=np.array([1.05, 0.0]),
        ts=2.0,
    )
    second = field._score_primary_actions(second_anchor, pred_mu=np.zeros(2, dtype=np.float64))
    assert second.action == "ASSIMILATE"
    assert second.trace_id == trace_id
    assert field._apply_primary_action(second, second_anchor) == trace_id

    trace = field.traces[trace_id]
    trace.evidence = 10.0
    trace.stability = 0.8
    trace.z_sigma_diag[:] = 0.05
    trace.ctx_sigma_diag[:] = 0.25

    split_decision = field._score_primary_actions(
        Anchor(
            anchor_id="a3",
            packet_id="p3",
            session_id="session-a",
            turn_id="turn-3",
            source="user",
            z=np.array([4.0] + [0.0] * (field.config.latent_dim - 1)),
            ts=3.0,
        ),
        pred_mu=np.zeros(field.config.latent_dim, dtype=np.float64),
    )
    assert split_decision.action == "SPLIT"
    assert split_decision.trace_id == trace_id

    trace.ctx_sigma_diag[:] = 0.05
    birth_decision = field._score_primary_actions(
        Anchor(
            anchor_id="a4",
            packet_id="p4",
            session_id="session-a",
            turn_id="turn-4",
            source="user",
            z=np.array([0.0, 1.0] + [0.0] * (field.config.latent_dim - 2)),
            ts=4.0,
        ),
        pred_mu=np.zeros(field.config.latent_dim, dtype=np.float64),
    )
    assert birth_decision.action == "BIRTH"


def test_posterior_slice_and_workspace_follow_context(field_factory: FieldFactory) -> None:
    del field_factory
    field = AuroraField(FieldConfig(latent_dim=2, context_dim=2, candidate_size=8, workspace_size=2))
    left = field._new_trace(
        Anchor(
            anchor_id="left-a",
            packet_id="left-p",
            session_id="session-a",
            turn_id="turn-left",
            source="user",
            z=np.array([1.0, 0.0]),
            ts=0.0,
        ),
        np.array([1.0, 0.0]),
        trace_id="left",
    )
    right = field._new_trace(
        Anchor(
            anchor_id="right-a",
            packet_id="right-p",
            session_id="session-a",
            turn_id="turn-right",
            source="user",
            z=np.array([1.0, 0.0]),
            ts=0.0,
        ),
        np.array([0.0, 1.0]),
        trace_id="right",
    )
    left.evidence = 5.0
    right.evidence = 5.0
    field.trace_store.add(left)
    field.trace_store.add(right)
    group = field._create_or_extend_group("left", "right")

    left_slice = field._posterior_slice(group, np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    right_slice = field._posterior_slice(group, np.array([1.0, 0.0]), np.array([0.0, 1.0]))

    assert left_slice[0] > 0.75
    assert right_slice[1] > 0.75

    ws_left = field._settle_workspace(
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
        np.zeros(2, dtype=np.float64),
        "session-a",
    )
    ws_right = field._settle_workspace(
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.zeros(2, dtype=np.float64),
        "session-a",
    )

    assert ws_left.activation["left"] > 0.75
    assert ws_right.activation["right"] > 0.75


def test_workspace_readout_prefers_matching_session(field_factory: FieldFactory) -> None:
    field = field_factory()
    field.inject(
        {
            "session_id": "session-a",
            "turn_id": "turn-1",
            "source": "user",
            "payload": "Remember the note.",
            "meta": {"role": "user"},
        }
    )
    field.inject(
        {
            "session_id": "session-b",
            "turn_id": "turn-2",
            "source": "user",
            "payload": "Remember the note.",
            "meta": {"role": "user"},
        }
    )

    workspace = field.read_workspace({"payload": "Remember the note.", "session_id": "session-a"}, k=1)
    trace = field.trace_store.get(workspace.active_trace_ids[0])

    assert trace is not None
    assert trace.metadata["session_id"] == "session-a"
    assert workspace.metadata["session_id"] == "session-a"


def test_inhibitory_edges_reduce_activation_during_settle(field_factory: FieldFactory) -> None:
    del field_factory
    field = AuroraField(FieldConfig(latent_dim=2, context_dim=2, candidate_size=8, workspace_size=2))
    left = field._new_trace(
        Anchor(
            anchor_id="a",
            packet_id="p",
            session_id="session-a",
            turn_id="turn-a",
            source="user",
            z=np.array([1.0, 0.0]),
            ts=1.0,
        ),
        np.array([1.0, 0.0]),
        trace_id="trace-a",
    )
    right = field._new_trace(
        Anchor(
            anchor_id="b",
            packet_id="q",
            session_id="session-a",
            turn_id="turn-b",
            source="user",
            z=np.array([1.0, 0.0]),
            ts=2.0,
        ),
        np.array([1.0, 0.0]),
        trace_id="trace-b",
    )
    left.evidence = 5.0
    left.access_ema = 2.0
    right.evidence = 1.0
    field.trace_store.add(left)
    field.trace_store.add(right)
    field._upsert_edge("trace-a", "trace-b", "inhib", delta_weight=1.0, ts=3.0)

    workspace = field._settle_workspace(
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
        np.zeros(2, dtype=np.float64),
        "session-a",
    )

    assert workspace.active_trace_ids[0] == "trace-a"
    if len(workspace.weights) > 1:
        assert workspace.weights[0] > workspace.weights[1]
