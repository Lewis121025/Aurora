from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from typing import Any, cast

import pytest

from aurora.runtime.contracts import AffectiveContent, EpisodeContent, EvidenceContent, MemoryAtom, SemanticContent
from aurora.runtime.projections import build_memory_projection
from tests.conftest import KernelFactory, QueueLLM, scripted_memory_llm


def _normalize(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _normalize(item) for key, item in asdict(cast(Any, value)).items()}
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def _json_text(value: object) -> str:
    return json.dumps(_normalize(value), ensure_ascii=False, sort_keys=True)


def _kernel(kernel_factory: KernelFactory, *steps: str) -> Any:
    return cast(
        Any,
        kernel_factory(
            llm=QueueLLM(
                *(steps or ("ack",)),
                repeat_last=True,
                structured=scripted_memory_llm,
            )
        ),
    )


def _stored_atoms(kernel: Any, subject_id: str, *, atom_kind: str | None = None) -> list[MemoryAtom]:
    atoms = list(kernel.store.list_atoms(subject_id))
    if atom_kind is None:
        return atoms
    return [atom for atom in atoms if atom.atom_kind == atom_kind]


def _semantic_values(items: tuple[Any, ...]) -> set[str]:
    return {item.value for item in items}


def test_evidence_becomes_stateful_memory_and_keeps_evidence_archive(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-evidence"

    turn = kernel.turn(subject_id, "我在杭州工作，也喜欢爵士乐，今天心情很好。", now_ts=1.0)
    state = kernel.state(subject_id)
    stored_atoms = _stored_atoms(kernel, subject_id)

    assert turn.subject_id == subject_id
    assert turn.response_text.strip() == "ack"
    assert turn.recall_used is True
    assert state.subject_id == subject_id
    assert _semantic_values(state.semantic_self_model) == {"杭州", "爵士乐"}
    assert state.affective_state.mood == "positive"
    assert state.recent_episodes
    assert state.recent_episodes[-1].summary
    assert state.recent_episodes[-1].actors
    assert state.recent_episodes[-1].time_span.start == 1.0
    assert any(isinstance(atom.content, EvidenceContent) for atom in stored_atoms)
    assert any(atom.atom_kind == "semantic" for atom in stored_atoms)
    assert any(atom.atom_kind == "affective" for atom in stored_atoms)


def test_public_state_is_projection_only(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-projection-only"

    state = kernel.state(subject_id)
    empty_state_json = _json_text(state)
    assert "atom_id" not in empty_state_json
    assert "source_atom_ids" not in empty_state_json
    assert "supersedes_atom_id" not in empty_state_json
    assert "inhibits_atom_ids" not in empty_state_json

    kernel.turn(subject_id, "我在杭州工作，也喜欢爵士乐。", now_ts=1.0)
    state = kernel.state(subject_id)
    state_json = _json_text(state)

    assert all(not isinstance(item, MemoryAtom) for item in state.semantic_self_model)
    assert all(not isinstance(item, MemoryAtom) for item in state.procedural_memory)
    assert all(not isinstance(item, MemoryAtom) for item in state.recent_episodes)
    assert "atom_id" not in state_json
    assert "source_atom_ids" not in state_json
    assert "supersedes_atom_id" not in state_json
    assert "inhibits_atom_ids" not in state_json


def test_memory_atom_payloads_stay_typed(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-typed-payloads"

    kernel.turn(subject_id, "我在杭州工作，也喜欢爵士乐，今天心情很好。", now_ts=1.0)

    stored_atoms = _stored_atoms(kernel, subject_id)
    assert all(not isinstance(atom.content, dict) for atom in stored_atoms)
    assert any(isinstance(atom.content, EvidenceContent) for atom in stored_atoms)
    assert any(isinstance(atom.content, EpisodeContent) for atom in stored_atoms)
    assert any(isinstance(atom.content, SemanticContent) for atom in stored_atoms)
    assert any(isinstance(atom.content, AffectiveContent) for atom in stored_atoms)


def test_response_reads_user_compiled_state_before_generation(kernel_factory: KernelFactory) -> None:
    llm = QueueLLM(
        lambda messages: _assert_current_state_visible(messages),
        repeat_last=False,
        structured=scripted_memory_llm,
    )
    kernel = cast(Any, kernel_factory(llm=llm))
    subject_id = "subject-pre-response-compile"

    kernel.turn(subject_id, "我现在住在杭州。", now_ts=1.0)


def _assert_current_state_visible(messages: list[dict[str, str]]) -> str:
    state_segment = messages[1]["content"]
    assert "杭州" in state_segment
    assert "上海" not in state_segment
    return "ack"


def test_recall_requires_explicit_temporal_scope(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    with pytest.raises(TypeError):
        kernel.recall("subject", "我住在哪里？")


def test_semantic_reconsolidation_supersedes_without_silent_overwrite(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-reconsolidation"

    kernel.turn(subject_id, "我住在上海。", now_ts=1.0)
    kernel.turn(subject_id, "更正，我现在住在杭州。", now_ts=2.0)

    state = kernel.state(subject_id)
    stored_semantic = _stored_atoms(kernel, subject_id, atom_kind="semantic")

    assert _semantic_values(state.semantic_self_model) == {"杭州"}
    assert any(atom.status == "superseded" and "上海" in _json_text(atom.content) for atom in stored_semantic)
    assert any(atom.status == "active" and "杭州" in _json_text(atom.content) for atom in stored_semantic)
    assert "我住在杭州" in _json_text(state.active_cognition)
    assert "我住在上海" not in _json_text(state.active_cognition)

    current = kernel.recall(subject_id, "我现在住在哪里？", temporal_scope="current", limit=5, mode="blended")
    assert current.hits
    assert any(hit.memory_kind == "semantic" and "杭州" in hit.content for hit in current.hits)
    assert all("上海" not in hit.content for hit in current.hits if hit.memory_kind == "semantic")

    historical = kernel.recall(subject_id, "我以前住在哪里？", temporal_scope="historical", limit=10, mode="blended")
    assert any("上海" in hit.content for hit in historical.hits)
    assert all("杭州" not in hit.content for hit in historical.hits if hit.memory_kind == "semantic")


def test_inhibition_hides_memory_without_deleting_evidence(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-inhibition"

    kernel.turn(subject_id, "我喜欢爵士乐。", now_ts=1.0)
    kernel.turn(subject_id, "请忘掉我喜欢爵士乐。", now_ts=2.0)

    state = kernel.state(subject_id)
    stored_atoms = _stored_atoms(kernel, subject_id)

    assert "爵士乐" not in _json_text(state.semantic_self_model)
    assert any(atom.atom_kind == "inhibition" for atom in stored_atoms)
    assert any(atom.atom_kind == "evidence" and "我喜欢爵士乐" in _json_text(atom.content) for atom in stored_atoms)
    assert any(atom.atom_kind == "evidence" and "请忘掉我喜欢爵士乐" in _json_text(atom.content) for atom in stored_atoms)

    recalled = kernel.recall(subject_id, "我喜欢什么音乐？", temporal_scope="current", limit=5, mode="blended")
    assert all("爵士乐" not in hit.content for hit in recalled.hits)


def test_inhibited_content_does_not_reenter_projection(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-inhibition-projection"

    kernel.turn(subject_id, "我喜欢爵士乐。", now_ts=1.0)
    kernel.turn(subject_id, "请忘掉我喜欢爵士乐。", now_ts=2.0)

    state_segment, episode_segment, _ = build_memory_projection(kernel.state(subject_id), ())

    assert "爵士乐" not in state_segment
    assert "爵士乐" not in episode_segment


def test_reconsolidation_does_not_hide_unrelated_episode_context(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-episode-continuity"

    kernel.turn(subject_id, "我住在上海。今天和朋友吃饭很开心。", now_ts=1.0)
    kernel.turn(subject_id, "更正，我现在住在杭州。", now_ts=2.0)

    state = kernel.state(subject_id)

    assert any("朋友吃饭很开心" in episode.summary for episode in state.recent_episodes)
    assert any("更正，我现在住在杭州" in episode.summary for episode in state.recent_episodes)


def test_affective_episode_markers_and_current_mood_are_separate(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-affect"

    kernel.turn(subject_id, "今天见到老朋友，我很开心，也有点放松。", now_ts=1.0)

    state = kernel.state(subject_id)
    episode = state.recent_episodes[-1]

    assert episode.emotion_markers
    assert state.affective_state.mood == "positive"
    assert "开心" in state.affective_state.active_feelings
    assert _json_text(episode.emotion_markers) != _json_text(state.affective_state)


def test_cognitive_summary_stays_structured_without_raw_thought(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-cognition"

    kernel.turn(subject_id, "我想先把部署文档补完，再同步团队，不想让大家一直等。", now_ts=1.0)

    cognition_text = _json_text(kernel.state(subject_id).active_cognition)

    assert "raw_thought" not in cognition_text
    assert "reasoning" not in cognition_text
    assert "scratchpad" not in cognition_text
    for key in ("beliefs", "goals", "conflicts", "intentions", "commitments"):
        assert key in cognition_text


def test_cognitive_snapshot_is_single_active_state(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-cognition-snapshot"

    kernel.turn(subject_id, "我想先把部署文档补完。", now_ts=1.0)
    kernel.turn(subject_id, "我想先去散步。", now_ts=2.0)

    state = kernel.state(subject_id)
    stored_cognition = _stored_atoms(kernel, subject_id, atom_kind="cognitive")

    assert "先去散步" in _json_text(state.active_cognition)
    assert "先把部署文档补完" not in _json_text(state.active_cognition)
    assert sum(1 for atom in stored_cognition if atom.status == "active") == 1
    assert sum(1 for atom in stored_cognition if atom.status == "superseded") == 1
    assert any(atom.supersedes_atom_id for atom in stored_cognition if atom.status == "active")


def test_procedural_plan_is_single_active_snapshot(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-procedural-plan-snapshot"

    kernel.turn(subject_id, "我想先把部署文档补完，再同步团队。", now_ts=1.0)
    kernel.turn(subject_id, "我想先去散步，再回来写文档。", now_ts=2.0)

    state = kernel.state(subject_id)
    stored_procedures = _stored_atoms(kernel, subject_id, atom_kind="procedural")

    assert len(state.procedural_memory) == 1
    assert "先去散步，再回来写文档" in state.procedural_memory[-1].text
    assert "先把部署文档补完，再同步团队" not in _json_text(state.procedural_memory)
    assert sum(1 for atom in stored_procedures if atom.status == "active" and "plan" in _json_text(atom.content)) == 1
    assert any(atom.supersedes_atom_id for atom in stored_procedures if atom.status == "active" and "plan" in _json_text(atom.content))


def test_assistant_commitment_enters_procedural_memory(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory, "我会明天提醒你同步团队。")
    subject_id = "subject-assistant-commitment"

    kernel.turn(subject_id, "明天提醒我同步团队。", now_ts=1.0)

    state = kernel.state(subject_id)
    recalled = kernel.recall(subject_id, "提醒我同步团队", temporal_scope="current", limit=5, mode="blended")

    assert any("Aurora承诺：明天提醒你同步团队" in item.text for item in state.procedural_memory)
    assert any(item.trigger == "assistant_commitment" for item in state.procedural_memory)
    assert any("Aurora承诺：明天提醒你同步团队" in hit.content for hit in recalled.hits)


def test_narrative_memory_accumulates_across_episodes(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-narrative"

    kernel.turn(subject_id, "搬到杭州以后，我在重新适应工作节奏。", now_ts=1.0)
    kernel.turn(subject_id, "最近我把周末留给朋友和运动，生活开始重新排好。", now_ts=2.0)

    state = kernel.state(subject_id)

    assert state.narrative_state.arcs
    assert any(arc.episode_count >= 2 for arc in state.narrative_state.arcs)
    assert any("杭州" in arc.theme or "杭州" in arc.storyline for arc in state.narrative_state.arcs)


def test_episode_captures_both_sides_of_the_turn(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory, "我会明天提醒你同步团队。")
    subject_id = "subject-episode-full-turn"

    kernel.turn(subject_id, "明天提醒我同步团队。", now_ts=1.0)

    episode = kernel.state(subject_id).recent_episodes[-1]
    assert "明天提醒我同步团队" in episode.summary
    assert "我会明天提醒你同步团队" in episode.summary
    assert episode.time_span.start < episode.time_span.end


def test_blended_recall_surfaces_episode_semantic_and_narrative_memories(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-blended-recall"

    kernel.turn(subject_id, "我在杭州工作，也喜欢爵士乐。", now_ts=1.0)
    kernel.turn(subject_id, "搬到杭州以后，我在重新适应工作节奏。", now_ts=2.0)
    kernel.turn(subject_id, "最近我把周末留给朋友和运动，生活开始重新排好。", now_ts=3.0)

    recalled = kernel.recall(subject_id, "杭州 适应 生活", temporal_scope="current", limit=10, mode="blended")

    assert any(hit.memory_kind == "episode" for hit in recalled.hits)
    assert any(hit.memory_kind == "semantic" for hit in recalled.hits)
    assert any(hit.memory_kind == "narrative" for hit in recalled.hits)


def test_both_scope_recall_dedupes_publicly_indistinguishable_hits(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-recall-dedupe"

    kernel.turn(subject_id, "我住在杭州。", now_ts=1.0)
    kernel.turn(subject_id, "更正，我现在住在上海。", now_ts=2.0)
    kernel.turn(subject_id, "更正，我现在住在杭州。", now_ts=3.0)

    recalled = kernel.recall(subject_id, "我住在哪里？", temporal_scope="both", limit=10, mode="blended")
    semantic_contents = [hit.content for hit in recalled.hits if hit.memory_kind == "semantic"]

    assert semantic_contents.count("我现在住在杭州") == 1


def test_affective_atoms_are_single_active_snapshot(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-affective-snapshot"

    kernel.turn(subject_id, "今天很开心。", now_ts=1.0)
    kernel.turn(subject_id, "现在有点焦虑。", now_ts=2.0)

    state = kernel.state(subject_id)
    stored_affective = _stored_atoms(kernel, subject_id, atom_kind="affective")

    assert "焦虑" in _json_text(state.affective_state)
    assert "开心" not in _json_text(state.affective_state)
    assert sum(1 for atom in stored_affective if atom.status == "active") == 1
    assert sum(1 for atom in stored_affective if atom.status == "superseded") == 1
    assert any(atom.supersedes_atom_id for atom in stored_affective if atom.status == "active")
