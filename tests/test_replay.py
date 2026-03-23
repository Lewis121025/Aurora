from __future__ import annotations

import numpy as np

from aurora.core.types import Anchor, ExperienceFrame, PosteriorGroup
from aurora.core.config import FieldConfig
from aurora.runtime import AuroraField
from aurora.runtime.proposals import action_vector

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

    procedure_support = [trace.role_support.get("procedure", 0.0) for trace in field.traces.values()]
    assert max(procedure_support) > 0.0
    option_edges = [edge for edge in field.edges.values() if edge.kind == "option"]
    assert option_edges


def test_replay_batch_scales_down_under_tight_maintenance_budget(field_factory: FieldFactory) -> None:
    field = field_factory()
    for index, payload in enumerate(["left", "right"] * 12, start=1):
        field.inject(
            {
                "payload": payload,
                "session_id": "session-a",
                "turn_id": f"turn-{index}",
                "source": "user",
            }
        )

    field.rng = np.random.default_rng(0)
    low_budget = field._sample_replay_batch(ms_budget=8)
    field.rng = np.random.default_rng(0)
    high_budget = field._sample_replay_batch(ms_budget=24)

    assert low_budget
    assert high_budget
    assert len(low_budget) <= len(high_budget)
    assert len(low_budget) < field.replay_config.batch_size


def test_role_lifecycle_demotes_unsupported_role(field_factory: FieldFactory) -> None:
    field = field_factory()
    result = field.inject(
        {
            "payload": "prototype seed",
            "session_id": "session-a",
            "turn_id": "turn-1",
            "source": "user",
        }
    )
    trace = field.traces[result.trace_ids[0]]
    trace.role_logits["prototype"] = 0.15
    trace.role_support["prototype"] = 0.0
    trace.role_gain_ema["prototype"] = 0.0
    field.budget_config.max_traces = 1
    field.trace_store.add(trace)

    demoted = field._role_lifecycle_step()

    assert trace.trace_id in demoted
    assert trace.role_logits["prototype"] == 0.0


def test_replay_reconsolidation_distributes_credit_by_activation(field_factory: FieldFactory) -> None:
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
            ts=1.0,
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
            z=np.array([0.8, 0.2], dtype=np.float64),
            ts=1.0,
        ),
        np.array([0.8, 0.2], dtype=np.float64),
        trace_id="right",
    )
    field.trace_store.add(left)
    field.trace_store.add(right)
    field.ann_index.add_or_update(left)
    field.ann_index.add_or_update(right)

    frame = ExperienceFrame(
        frame_id="f0",
        anchor_id="left-a",
        trace_id="left",
        session_id="session-a",
        turn_id="turn-1",
        action="ASSIMILATE",
        x=np.array([1.0, 0.0], dtype=np.float64),
        context=np.array([1.0, 0.0], dtype=np.float64),
        workspace_vec=np.array([1.0, 0.0], dtype=np.float64),
        frontier_vec=np.array([1.0, 0.0], dtype=np.float64),
        action_vec=action_vector("ASSIMILATE"),
        ts=2.0,
        active_ids=("left", "right"),
        activation={"left": 0.6, "right": 0.4},
        next_x=np.array([0.9, 0.1], dtype=np.float64),
        next_trace_id="right",
    )

    replayed = field._reconsolidate_batch([frame])

    assert "left" in replayed
    assert "right" in replayed
    assert field.traces["left"].evidence > 1.0
    assert field.traces["right"].evidence > 1.0


def test_role_mutation_is_accepted_only_when_delta_is_negative(field_factory: FieldFactory) -> None:
    field = field_factory()
    result = field.inject(
        {
            "payload": "role candidate",
            "session_id": "session-a",
            "turn_id": "turn-1",
            "source": "user",
        }
    )
    trace = field.traces[result.trace_ids[0]]
    trace.role_support["prototype"] = 3.5
    trace.role_gain_ema["prototype"] = 1.2

    accepted = field._accept_role_mutation(trace, role="prototype", member_ids=("m1", "m2"))

    assert accepted is True
    assert trace.role_logits["prototype"] > 0.0
    assert trace.metadata["last_role_delta"] < 0.0


def test_replay_trace_mutation_spawns_structural_trace_and_updates_objective() -> None:
    field = AuroraField(FieldConfig(latent_dim=2, context_dim=2, candidate_size=8, workspace_size=4))
    anchor = Anchor(
        anchor_id="replay-a",
        packet_id="replay-p",
        session_id="session-a",
        turn_id="turn-1",
        source="user",
        z=np.array([3.0, 0.0], dtype=np.float64),
        ts=2.0,
    )
    field.anchor_store.add_anchor(anchor)
    base = field._new_trace(
        Anchor(
            anchor_id="base-a",
            packet_id="base-p",
            session_id="session-a",
            turn_id="turn-0",
            source="user",
            z=np.array([0.0, 0.0], dtype=np.float64),
            ts=1.0,
        ),
        np.array([0.0, 1.0], dtype=np.float64),
        trace_id="base",
    )
    base.evidence = 8.0
    base.stability = 0.9
    base.z_sigma_diag[:] = 0.05
    base.ctx_sigma_diag[:] = 0.10
    field.trace_store.add(base)
    field.ann_index.add_or_update(base)

    frame = ExperienceFrame(
        frame_id="frame-replay",
        anchor_id=anchor.anchor_id,
        trace_id="base",
        session_id="session-a",
        turn_id="turn-1",
        action="ASSIMILATE",
        x=np.array([3.0, 0.0], dtype=np.float64),
        context=np.array([0.0, 1.0], dtype=np.float64),
        workspace_vec=np.array([0.0, 1.0], dtype=np.float64),
        frontier_vec=np.array([0.0, 1.0], dtype=np.float64),
        action_vec=action_vector("ASSIMILATE"),
        ts=2.0,
        active_ids=("base",),
        activation={"base": 1.0},
        next_x=np.array([3.1, 0.0], dtype=np.float64),
        next_trace_id="base",
    )

    created = field._replay_trace_mutations([frame])

    assert len(created) == 1
    spawned = field.traces[created[0]]
    assert spawned.metadata["spawned_from_frame"] == frame.frame_id
    assert spawned.metadata["source_anchor_ids"] == [anchor.anchor_id]
    assert spawned.metadata["spawn_mode"] in {"replay_birth", "replay_split"}

    terms = field._replay_structural_objective(
        [frame],
        predictor_loss=0.0,
        replayed_group_ids=(),
        structural_trace_ids=created,
        compressed_trace_ids=(),
        demoted_trace_ids=(),
        pruned_trace_ids=(),
    )

    assert terms.extras["structural_trace_count"] == 1.0
    assert terms.extras["structural_churn"] > 0.0


def test_continuation_stats_update_traces_groups_and_maintenance_future_costs() -> None:
    field = AuroraField(FieldConfig(latent_dim=2, context_dim=2, candidate_size=8, workspace_size=4))
    left = field._new_trace(
        Anchor(
            anchor_id="left-a",
            packet_id="left-p",
            session_id="session-a",
            turn_id="turn-left",
            source="user",
            z=np.array([1.0, 0.0], dtype=np.float64),
            ts=1.0,
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
            z=np.array([0.8, 0.2], dtype=np.float64),
            ts=1.2,
        ),
        np.array([0.8, 0.2], dtype=np.float64),
        trace_id="right",
    )
    left.posterior_group_ids = ("g0",)
    right.posterior_group_ids = ("g0",)
    field.trace_store.add(left)
    field.trace_store.add(right)
    field.ann_index.add_or_update(left)
    field.ann_index.add_or_update(right)
    field.groups["g0"] = PosteriorGroup(
        group_id="g0",
        member_ids=["left", "right"],
        alpha=np.array([2.0, 1.5, 0.2], dtype=np.float64),
        ctx_mu=np.array([[1.0, 0.0], [0.7, 0.3], [0.0, 0.0]], dtype=np.float64),
        ctx_sigma_diag=np.full((3, 2), 0.2, dtype=np.float64),
        pred_success_ema=np.array([0.8, 0.7, 0.5], dtype=np.float64),
        unresolved_mass=0.3,
        temperature=1.15,
    )

    frame = ExperienceFrame(
        frame_id="frame-cont",
        anchor_id="left-a",
        trace_id="left",
        session_id="session-a",
        turn_id="turn-1",
        action="ASSIMILATE",
        x=np.array([1.0, 0.0], dtype=np.float64),
        context=np.array([1.0, 0.0], dtype=np.float64),
        workspace_vec=np.array([1.0, 0.0], dtype=np.float64),
        frontier_vec=np.array([1.0, 0.0], dtype=np.float64),
        action_vec=action_vector("ASSIMILATE"),
        ts=2.0,
        active_ids=("left", "right"),
        activation={"left": 0.65, "right": 0.35},
        group_entropy=0.4,
        next_x=np.array([0.9, 0.1], dtype=np.float64),
        next_trace_id="right",
    )

    field._update_continuation_stats([frame])

    assert field.traces["left"].future_alignment_ema > 0.0
    assert field.traces["left"].future_drift_ema > 0.0
    assert field.traces["right"].future_alignment_ema > 0.0
    assert field.groups["g0"].replay_alignment_ema > 0.0
    assert field.groups["g0"].replay_tension_ema > 0.0

    terms = field._replay_structural_objective(
        [frame],
        predictor_loss=0.0,
        replayed_group_ids=("g0",),
        structural_trace_ids=(),
        compressed_trace_ids=(),
        demoted_trace_ids=(),
        pruned_trace_ids=(),
    )

    assert terms.extras["maintenance_future_alignment"] > 0.0
    assert terms.extras["maintenance_future_drift"] > 0.0
    assert terms.extras["group_heat"] > 0.0
