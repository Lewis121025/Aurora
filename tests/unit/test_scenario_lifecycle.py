"""场景级生命周期回归测试。

模拟多轮交互 → doze → sleep → 恢复的完整流程，
验证跨模块不变量：记忆图完整性、关系投影、定向拓扑、持久化往返。
"""
from __future__ import annotations

from pathlib import Path

from aurora.evaluation.continuity import evaluate_continuity
from aurora.evaluation.relation_dynamics import evaluate_relation_dynamics
from aurora.evaluation.sleep_effects import evaluate_sleep_effects, snapshot_sleep_state
from aurora.relation.projectors import project_relation
from aurora.runtime.contracts import Phase
from aurora.runtime.engine import AuroraEngine

from tests.conftest import ContextAwareLLM, StubLLM


def _build_multi_turn_engine(tmp_path: Path) -> AuroraEngine:
    engine = AuroraEngine.create(data_dir=str(tmp_path), llm=ContextAwareLLM())
    engine.handle_turn(session_id="s1", text="谢谢你理解我，我很感动")
    engine.handle_turn(session_id="s1", text="我受伤了，需要一些空间")
    engine.handle_turn(session_id="s1", text="对不起，我想修复我们的关系")
    engine.handle_turn(session_id="s2", text="你好，这是第二段关系")
    return engine


def test_full_lifecycle_awake_doze_sleep(tmp_path: Path) -> None:
    engine = _build_multi_turn_engine(tmp_path)

    assert engine.state.metabolic.phase is Phase.AWAKE
    assert len(engine.memory_store.fragments) >= 8
    assert engine.relation_store.moment_count() >= 4

    pre_snap = snapshot_sleep_state(engine.state, engine.memory_store)

    engine.doze()
    assert engine.state.metabolic.phase is Phase.DOZE

    engine.sleep()
    assert engine.state.metabolic.phase is Phase.SLEEP
    assert engine.memory_store.sleep_cycles >= 1
    assert engine.state.metabolic.pending_sleep_relation_ids == ()

    post_snap = snapshot_sleep_state(engine.state, engine.memory_store)
    effects = evaluate_sleep_effects(pre_snap, post_snap)
    assert effects.ok


def test_continuity_after_full_lifecycle(tmp_path: Path) -> None:
    engine = _build_multi_turn_engine(tmp_path)
    engine.doze()
    engine.sleep()

    check = evaluate_continuity(engine.state, engine.memory_store, engine.relation_store)
    assert check.ok, (
        f"active_relations_known={check.active_relations_known}, "
        f"active_knots_known={check.active_knots_known}, "
        f"anchor_threads_known={check.anchor_threads_known}, "
        f"transitions_monotonic={check.transitions_monotonic}"
    )


def test_relation_dynamics_after_mixed_turns(tmp_path: Path) -> None:
    engine = _build_multi_turn_engine(tmp_path)
    engine.sleep()

    relation_id = engine.identity.relation_for("s1")
    assert relation_id is not None
    dynamics = evaluate_relation_dynamics(
        engine.relation_store, engine.memory_store, relation_id,
    )
    assert dynamics.ok
    assert dynamics.moment_count >= 3
    assert dynamics.relation_has_memory


def test_relation_projection_reflects_history(tmp_path: Path) -> None:
    engine = _build_multi_turn_engine(tmp_path)

    relation_id = engine.identity.relation_for("s1")
    assert relation_id is not None
    formation = engine.relation_store.formation_for(relation_id)
    projection = project_relation(formation, 1_000_000.0)

    assert 0.0 <= projection.trust <= 1.0
    assert 0.0 <= projection.distance <= 1.0
    assert 0.0 <= projection.warmth <= 1.0
    assert projection.trust > 0.0 or projection.warmth > 0.0


def test_orientation_absorbs_topology_from_sleep(tmp_path: Path) -> None:
    engine = _build_multi_turn_engine(tmp_path)
    engine.sleep()

    orientation = engine.state.orientation
    all_sources = (
        list(orientation.self_evidence.get("recognition", ()))
        + list(orientation.self_evidence.get("agency", ()))
        + list(orientation.self_evidence.get("fragility", ()))
        + list(orientation.world_evidence.get("stability", ()))
        + list(orientation.world_evidence.get("risk", ()))
    )
    has_topology_sources = any(
        s.startswith("thread_") or s.startswith("knot_")
        for s in all_sources
    )
    assert has_topology_sources, "Orientation should contain topology-derived evidence after sleep"


def test_persistence_roundtrip_preserves_full_state(tmp_path: Path) -> None:
    first = _build_multi_turn_engine(tmp_path)
    first.doze()
    first.sleep()

    frag_count = len(first.memory_store.fragments)
    thread_count = len(first.memory_store.threads)
    moment_count = first.relation_store.moment_count()
    turn_count = first.persistence.turn_count()

    second = AuroraEngine.create(data_dir=str(tmp_path), llm=StubLLM())

    assert len(second.memory_store.fragments) == frag_count
    assert len(second.memory_store.threads) == thread_count
    assert second.relation_store.moment_count() == moment_count
    assert second.persistence.turn_count() == turn_count
    assert second.memory_store.sleep_cycles >= 1

    check = evaluate_continuity(second.state, second.memory_store, second.relation_store)
    assert check.ok


def test_memory_graph_integrity_after_sediment(tmp_path: Path) -> None:
    engine = _build_multi_turn_engine(tmp_path)
    engine.sleep()

    for fid, fragment in engine.memory_store.fragments.items():
        for tid in fragment.thread_ids:
            assert tid in engine.memory_store.threads, f"Fragment {fid} references missing thread {tid}"
        for kid in fragment.knot_ids:
            assert kid in engine.memory_store.knots, f"Fragment {fid} references missing knot {kid}"

    for eid, edge in engine.memory_store.associations.items():
        assert edge.src_fragment_id in engine.memory_store.fragments, f"Edge {eid} missing src"
        assert edge.dst_fragment_id in engine.memory_store.fragments, f"Edge {eid} missing dst"

    for tid, trace in engine.memory_store.traces.items():
        assert trace.fragment_id in engine.memory_store.fragments, f"Trace {tid} missing fragment"


def test_second_sleep_is_stable(tmp_path: Path) -> None:
    engine = _build_multi_turn_engine(tmp_path)
    engine.sleep()
    threads_1 = len(engine.memory_store.threads)
    knots_1 = len(engine.memory_store.knots)

    engine.sleep()
    assert len(engine.memory_store.threads) == threads_1
    assert len(engine.memory_store.knots) == knots_1
