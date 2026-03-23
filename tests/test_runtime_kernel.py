from __future__ import annotations

import numpy as np

from aurora.core.config import BudgetConfig, FieldConfig
from aurora.core.types import Anchor, PosteriorGroup, TraceRecord
from aurora.runtime import AuroraField
from aurora.runtime.objective_terms import (
    ObjectiveObservation,
    empirical_patch_terms,
    group_tension_cost,
    posterior_slice,
    objective_from_components,
    trace_fidelity_cost,
    trace_future_alignment_cost,
    trace_future_drift_cost,
    trace_role_cost,
    trace_storage_cost,
)

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
    assert np.isfinite(stats.objective_total)
    assert field.hot_trace_ids
    assert field.field_stats()["budget_state"]["max_traces"] == field.budget_config.max_traces
    assert field.field_stats()["objective"]["last"]["total"] == stats.objective_total
    assert field.field_stats()["objective"]["last"]["replay_activation_kl"] >= 0.0
    assert field.field_stats()["objective"]["last"]["replay_transition_gap"] >= 0.0
    assert field.field_stats()["objective"]["last"]["group_heat"] >= 0.0
    assert field.field_stats()["objective"]["last"]["structural_churn"] >= 0.0


def test_snapshot_round_trip_restores_v6_kernel_state(field_factory: FieldFactory) -> None:
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

    assert payload["schema_version"] == 6
    assert "objective_state" in payload
    assert restored.step == field.step
    assert len(restored.anchors) == len(field.anchors)
    assert len(restored.traces) == len(field.traces)
    assert len(restored.frames) == len(field.frames)
    assert restored.predictor.export_state().theta
    assert restored.field_stats()["objective"]["ema"] == field.field_stats()["objective"]["ema"]


def test_field_exposes_no_response_or_decoder_surface(field_factory: FieldFactory) -> None:
    field = field_factory()

    assert not hasattr(field, "respond")
    assert not hasattr(field, "local_decoder")
    assert not hasattr(field, "workspace_serializer")


def test_closed_loop_objective_total_is_consistent() -> None:
    config = FieldConfig(latent_dim=2, context_dim=2)
    terms = objective_from_components(
        config=config,
        surprise=1.0,
        storage=2.0,
        complexity=3.0,
        plasticity=0.5,
        slow_alignment=4.0,
        drift=1.5,
        fidelity=0.25,
        role=0.75,
        inhibit_bonus=-0.2,
    )
    expected_local = 1.0 + config.lambda_storage * 2.0 + config.lambda_complexity * 3.0 + config.lambda_plasticity * 0.5
    expected_total = (
        expected_local
        + config.lambda_predictor * 4.0
        + config.lambda_drift * 1.5
        + config.lambda_fidelity * 0.25
        + config.lambda_role * 0.75
        - 0.2
    )
    assert abs(terms.local_energy - expected_local) < 1e-9
    assert abs(terms.total - expected_total) < 1e-9


def test_role_and_fidelity_costs_are_positive_for_unstable_compressed_trace() -> None:
    trace = TraceRecord(
        trace_id="t0",
        z_mu=np.array([1.0, 0.0], dtype=np.float64),
        z_sigma_diag=np.array([0.2, 0.2], dtype=np.float64),
        ctx_mu=np.array([1.0, 0.0], dtype=np.float64),
        ctx_sigma_diag=np.array([0.3, 0.3], dtype=np.float64),
        evidence=3.0,
        stability=0.8,
        uncertainty=0.2,
        fidelity=0.4,
        compression_stage=2,
        future_alignment_ema=0.7,
        future_drift_ema=0.5,
        role_logits={"episodic": 0.6, "prototype": 1.2, "procedure": 0.0},
        role_support={"prototype": 0.4, "procedure": 0.0},
        role_gain_ema={"prototype": -0.1, "procedure": 0.0},
    )

    assert trace_fidelity_cost(trace) > 0.0
    assert trace_role_cost(trace) > 0.0
    assert trace_future_alignment_cost(trace) > 0.0
    assert trace_future_drift_cost(trace) > 0.0


def test_role_cost_rewards_better_alignment_with_supported_high_order_trace() -> None:
    trace = TraceRecord(
        trace_id="t1",
        z_mu=np.array([1.0, 0.0], dtype=np.float64),
        z_sigma_diag=np.array([0.2, 0.2], dtype=np.float64),
        ctx_mu=np.array([1.0, 0.0], dtype=np.float64),
        ctx_sigma_diag=np.array([0.3, 0.3], dtype=np.float64),
        evidence=3.0,
        stability=0.8,
        uncertainty=0.2,
        fidelity=1.0,
        role_logits={"episodic": 0.7, "prototype": 0.3, "procedure": 0.0},
        role_support={"prototype": 1.2, "procedure": 0.0},
        role_gain_ema={"prototype": 0.4, "procedure": 0.0},
    )
    baseline = trace_role_cost(trace)
    trace.role_logits["prototype"] = 1.3

    assert trace_role_cost(trace) < baseline


def test_storage_cost_drops_when_trace_is_compressed() -> None:
    config = FieldConfig(latent_dim=2, context_dim=2)
    trace = TraceRecord(
        trace_id="t0",
        z_mu=np.array([1.0, 0.0], dtype=np.float64),
        z_sigma_diag=np.array([0.2, 0.2], dtype=np.float64),
        ctx_mu=np.array([1.0, 0.0], dtype=np.float64),
        ctx_sigma_diag=np.array([0.3, 0.3], dtype=np.float64),
        evidence=3.0,
        stability=0.8,
        uncertainty=0.2,
        fidelity=1.0,
        compression_stage=0,
    )
    compressed = trace.clone()
    compressed.fidelity = 0.4
    compressed.compression_stage = 2

    assert trace_storage_cost(compressed, config) < trace_storage_cost(trace, config)


def test_group_tension_cost_tracks_replay_heat_and_unresolved_mass() -> None:
    group = PosteriorGroup(
        group_id="g0",
        member_ids=["a", "b"],
        alpha=np.array([1.0, 1.0, 0.5], dtype=np.float64),
        ctx_mu=np.zeros((3, 2), dtype=np.float64),
        ctx_sigma_diag=np.ones((3, 2), dtype=np.float64),
        pred_success_ema=np.array([0.5, 0.5, 0.5], dtype=np.float64),
        unresolved_mass=0.4,
        replay_tension_ema=0.6,
    )

    assert group_tension_cost(group) > 0.6


def test_empirical_patch_terms_reward_better_local_fit() -> None:
    config = FieldConfig(latent_dim=2, context_dim=2)
    left = TraceRecord(
        trace_id="left",
        z_mu=np.array([1.0, 0.0], dtype=np.float64),
        z_sigma_diag=np.array([0.15, 0.15], dtype=np.float64),
        ctx_mu=np.array([1.0, 0.0], dtype=np.float64),
        ctx_sigma_diag=np.array([0.2, 0.2], dtype=np.float64),
        evidence=4.0,
        stability=0.7,
        uncertainty=0.15,
    )
    right = TraceRecord(
        trace_id="right",
        z_mu=np.array([-1.0, 0.0], dtype=np.float64),
        z_sigma_diag=np.array([0.15, 0.15], dtype=np.float64),
        ctx_mu=np.array([-1.0, 0.0], dtype=np.float64),
        ctx_sigma_diag=np.array([0.2, 0.2], dtype=np.float64),
        evidence=4.0,
        stability=0.7,
        uncertainty=0.15,
    )
    traces = {"left": left, "right": right}
    good_terms, _ = empirical_patch_terms(
        config=config,
        candidate_ids=["left", "right"],
        traces=traces,
        edges={},
        groups={},
        observations=[ObjectiveObservation(x=np.array([0.95, 0.05]), c=np.array([1.0, 0.0]))],
    )
    bad_terms, _ = empirical_patch_terms(
        config=config,
        candidate_ids=["left", "right"],
        traces=traces,
        edges={},
        groups={},
        observations=[ObjectiveObservation(x=np.array([-0.95, 0.05]), c=np.array([1.0, 0.0]))],
    )

    assert good_terms.surprise < bad_terms.surprise


def test_generic_posterior_slice_respects_context() -> None:
    config = FieldConfig(latent_dim=2, context_dim=2)
    left = TraceRecord(
        trace_id="left",
        z_mu=np.array([1.0, 0.0], dtype=np.float64),
        z_sigma_diag=np.array([0.15, 0.15], dtype=np.float64),
        ctx_mu=np.array([1.0, 0.0], dtype=np.float64),
        ctx_sigma_diag=np.array([0.2, 0.2], dtype=np.float64),
        evidence=4.0,
        stability=0.7,
        uncertainty=0.15,
    )
    right = TraceRecord(
        trace_id="right",
        z_mu=np.array([1.0, 0.0], dtype=np.float64),
        z_sigma_diag=np.array([0.15, 0.15], dtype=np.float64),
        ctx_mu=np.array([-1.0, 0.0], dtype=np.float64),
        ctx_sigma_diag=np.array([0.2, 0.2], dtype=np.float64),
        evidence=4.0,
        stability=0.7,
        uncertainty=0.15,
    )
    group = PosteriorGroup(
        group_id="g0",
        member_ids=["left", "right"],
        alpha=np.array([1.0, 1.0, 0.2], dtype=np.float64),
        ctx_mu=np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]], dtype=np.float64),
        ctx_sigma_diag=np.full((3, 2), 0.2, dtype=np.float64),
        pred_success_ema=np.array([0.8, 0.8, 0.5], dtype=np.float64),
    )

    probs = posterior_slice(group, {"left": left, "right": right}, np.array([1.0, 0.0]), np.array([1.0, 0.0]), config)

    assert probs[0] > probs[1]


def test_fidelity_step_reduces_effective_budget_pressure() -> None:
    field = AuroraField(
        FieldConfig(latent_dim=2, context_dim=2),
        budget_config=BudgetConfig(max_traces=1, max_edges=16, max_groups=16),
    )
    for idx in range(2):
        trace = field._new_trace(
            Anchor(
                anchor_id=f"a-{idx}",
                packet_id=f"p-{idx}",
                session_id="session-a",
                turn_id=f"turn-{idx}",
                source="user",
                z=np.array([1.0, 0.0], dtype=np.float64),
                ts=float(idx + 1),
            ),
            np.array([1.0, 0.0], dtype=np.float64),
            trace_id=f"trace-{idx}",
        )
        field.trace_store.add(trace)
        field.ann_index.add_or_update(trace)

    before = field._budget_pressure()
    compressed, _ = field._fidelity_step()
    after = field._budget_pressure()

    assert before > 1.0
    assert compressed
    assert after < before
