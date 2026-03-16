from __future__ import annotations

from pathlib import Path

from aurora.memory.recall import recent_recall
from aurora.runtime.contracts import Phase
from aurora.runtime.engine import AuroraEngine

from tests.conftest import ContextAwareLLM, StubLLM


def test_awake_writes_canonical_graph_objects(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path), llm=StubLLM())

    output = engine.handle_turn(session_id="s1", text="谢谢你理解我，我有点害怕")

    assert output.turn_id
    assert output.response_text
    assert output.aurora_move in {"approach", "withhold", "boundary", "repair", "silence", "witness"}
    assert engine.state.metabolic.phase is Phase.AWAKE
    assert len(engine.memory_store.fragments) == 2
    assert len(engine.memory_store.traces) >= 2
    assert len(engine.memory_store.associations) >= 1
    assert engine.relation_store.moment_count() == 1
    assert engine.relation_store.relation_count() == 1
    relation_id = engine.identity.relation_for("s1")
    assert relation_id is not None
    assert engine.state.metabolic.pending_sleep_relation_ids == (relation_id,)


def test_boundary_input_pushes_boundary_move(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path), llm=ContextAwareLLM())

    output = engine.handle_turn(session_id="s2", text="不要再继续，停，边界在这里")

    assert output.aurora_move == "boundary"


def test_phase_transitions_are_lifecycle_events(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path), llm=StubLLM())
    engine.handle_turn(session_id="s3", text="我想知道我们之后会怎样")

    doze_output = engine.doze()
    sleep_output = engine.sleep()

    assert doze_output.phase is Phase.DOZE
    assert sleep_output.phase is Phase.SLEEP
    assert len(engine.state.transitions) >= 2
    assert engine.state.metabolic.pending_sleep_relation_ids == ()


def test_doze_hover_keeps_recent_relation_material_lightly_active(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path), llm=StubLLM())
    engine.handle_turn(session_id="s4", text="谢谢你理解我")

    engine.doze()

    relation_id = engine.identity.relation_for("s4")
    assert relation_id is not None
    recalled = recent_recall(engine.memory_store, relation_id, limit=4)
    assert recalled
    assert any(fragment.activation_count >= 1 for fragment in recalled)


def test_aurora_traces_differ_from_user_traces(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path), llm=StubLLM())
    engine.handle_turn(session_id="s5", text="谢谢你理解我，我很感动")

    fragments = list(engine.memory_store.fragments.values())
    assert len(fragments) == 2
    user_frag, aurora_frag = fragments[0], fragments[1]
    user_traces = engine.memory_store.traces_for_fragment(user_frag.fragment_id)
    aurora_traces = engine.memory_store.traces_for_fragment(aurora_frag.fragment_id)
    assert len(user_traces) >= 1
    assert len(aurora_traces) >= 1
    if user_traces and aurora_traces:
        user_intensities = {t.channel: t.intensity for t in user_traces}
        aurora_intensities = {t.channel: t.intensity for t in aurora_traces}
        shared = set(user_intensities) & set(aurora_intensities)
        if shared:
            differs = any(
                abs(user_intensities[ch] - aurora_intensities[ch]) > 1e-6
                or abs(user_traces[0].carry - aurora_traces[0].carry) > 1e-6
                for ch in shared
            )
            assert differs, "Aurora traces should be modulated differently from user traces"


def test_boundary_move_produces_weaker_association(tmp_path: Path) -> None:
    engine_boundary = AuroraEngine.create(data_dir=str(tmp_path / "b"), llm=ContextAwareLLM())
    engine_boundary.handle_turn(session_id="s", text="不要再继续，停，边界在这里")

    engine_approach = AuroraEngine.create(data_dir=str(tmp_path / "a"), llm=StubLLM())
    engine_approach.handle_turn(session_id="s", text="谢谢你理解我")

    boundary_assocs = list(engine_boundary.memory_store.associations.values())
    approach_assocs = list(engine_approach.memory_store.associations.values())
    assert boundary_assocs and approach_assocs
    assert boundary_assocs[0].weight < approach_assocs[0].weight
