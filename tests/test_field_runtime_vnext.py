from __future__ import annotations

import numpy as np

from aurora.core.types import Anchor, PredictorPeek
from aurora.core.config import FieldConfig
from aurora.runtime import AuroraField

from tests.conftest import FieldFactory


def _peek(dim: int) -> PredictorPeek:
    return PredictorPeek(
        mu=np.zeros(dim, dtype=np.float64),
        sigma_diag=np.ones(dim, dtype=np.float64),
        h=np.zeros(0, dtype=np.float64),
    )


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
    first = field._score_primary_actions(first_anchor, _peek(2))
    assert first.action == "BIRTH"
    assert first.objective_terms["total"] == first.delta_energy
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
    second = field._score_primary_actions(second_anchor, _peek(2))
    assert second.action == "ASSIMILATE"
    assert second.trace_id == trace_id
    assert "slow_alignment" in second.objective_terms
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
        _peek(field.config.latent_dim),
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
        _peek(field.config.latent_dim),
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

    assert ws_left.activation["left"] > ws_left.activation["right"]
    assert ws_right.activation["right"] > ws_right.activation["left"]
    assert ws_left.metadata["group_kl"] >= 0.0
    assert ws_right.metadata["group_kl"] >= 0.0


def test_objective_terms_include_group_regularization(field_factory: FieldFactory) -> None:
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
        np.array([-1.0, 0.0]),
        trace_id="right",
    )
    left.evidence = 5.0
    right.evidence = 5.0
    field.trace_store.add(left)
    field.trace_store.add(right)
    field.ann_index.add_or_update(left)
    field.ann_index.add_or_update(right)
    field._create_or_extend_group("left", "right")

    anchor = Anchor(
        anchor_id="query-a",
        packet_id="query-p",
        session_id="session-a",
        turn_id="turn-query",
        source="user",
        z=np.array([1.0, 0.0]),
        ts=1.0,
    )
    decision = field._score_primary_actions(anchor, _peek(2))

    assert decision.objective_terms["group_kl"] > 0.0


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


def test_unresolved_spawn_creates_trace_with_anchor_provenance(field_factory: FieldFactory) -> None:
    del field_factory
    field = AuroraField(FieldConfig(latent_dim=2, context_dim=2, unresolved_spawn_min_anchors=3))
    left = field._new_trace(
        Anchor(
            anchor_id="seed-a",
            packet_id="packet-a",
            session_id="session-a",
            turn_id="turn-a",
            source="user",
            z=np.array([1.0, 0.0]),
            ts=1.0,
        ),
        np.array([1.0, 0.0]),
        trace_id="left",
    )
    right = field._new_trace(
        Anchor(
            anchor_id="seed-b",
            packet_id="packet-b",
            session_id="session-a",
            turn_id="turn-b",
            source="user",
            z=np.array([1.0, 0.0]),
            ts=1.0,
        ),
        np.array([0.0, 1.0]),
        trace_id="right",
    )
    for trace in (left, right):
        field.trace_store.add(trace)
        field.ann_index.add_or_update(trace)
    group = field._create_or_extend_group("left", "right")

    anchors = []
    contexts = []
    for idx in range(3):
        anchor = Anchor(
            anchor_id=f"amb-{idx}",
            packet_id=f"packet-{idx}",
            session_id="session-a",
            turn_id=f"turn-{idx}",
            source="user",
            z=np.array([0.5, 0.5]) + 0.01 * idx,
            ts=2.0 + idx,
        )
        field.anchor_store.add_anchor(anchor)
        anchors.append(anchor.anchor_id)
        contexts.append(np.array([0.1, 0.1], dtype=np.float64))
    group.unresolved_mass = field.config.group_null_trigger + 0.1
    group.ambiguous_buffer.extend(anchors)
    group.ambiguous_ctx_buffer.extend(contexts)

    spawned_id = field._maybe_spawn_from_unresolved(group, ts=5.0)

    assert spawned_id is not None
    spawned = field.traces[spawned_id]
    assert spawned.metadata["spawn_mode"] == "unresolved_cluster"
    assert spawned.metadata["spawned_from_group"] == group.group_id
    assert spawned.metadata["source_anchor_ids"] == anchors


def test_option_edge_mutation_accepts_high_support_transition(field_factory: FieldFactory) -> None:
    del field_factory
    field = AuroraField(FieldConfig(latent_dim=2, context_dim=2))
    left = field._new_trace(
        Anchor(
            anchor_id="left-a",
            packet_id="left-p",
            session_id="session-a",
            turn_id="turn-left",
            source="user",
            z=np.array([1.0, 0.0], dtype=np.float64),
            ts=0.0,
        ),
        np.array([1.0, 0.0], dtype=np.float64),
        trace_id="left",
    )
    right = field._new_trace(
        Anchor(
            anchor_id="right-a",
            packet_id="right-p",
            session_id="session-a",
            turn_id="turn-right",
            source="user",
            z=np.array([0.9, 0.1], dtype=np.float64),
            ts=0.0,
        ),
        np.array([1.0, 0.0], dtype=np.float64),
        trace_id="right",
    )
    field.trace_store.add(left)
    field.trace_store.add(right)

    assert field._accept_option_edge_mutation("left", "right", proposed_support=0.95) is True


def test_pair_group_mutation_accepts_strong_ambiguous_pair(field_factory: FieldFactory) -> None:
    del field_factory
    field = AuroraField(FieldConfig(latent_dim=2, context_dim=2))
    left = field._new_trace(
        Anchor(
            anchor_id="left-a",
            packet_id="left-p",
            session_id="session-a",
            turn_id="turn-left",
            source="user",
            z=np.array([1.0, 0.0], dtype=np.float64),
            ts=0.0,
        ),
        np.array([1.0, 0.0], dtype=np.float64),
        trace_id="left",
    )
    right = field._new_trace(
        Anchor(
            anchor_id="right-a",
            packet_id="right-p",
            session_id="session-a",
            turn_id="turn-right",
            source="user",
            z=np.array([0.95, 0.05], dtype=np.float64),
            ts=0.0,
        ),
        np.array([1.0, 0.0], dtype=np.float64),
        trace_id="right",
    )
    left.evidence = 5.0
    right.evidence = 5.0
    field.trace_store.add(left)
    field.trace_store.add(right)

    assert field._accept_pair_group_mutation("left", "right", ambiguity=0.98, bf=1.5, ctx_overlap=0.95) is True
