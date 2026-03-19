from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from typing import Any, cast

from aurora.memory.experts import GLOBAL_EXPERT_ID
from aurora.memory.ledger import HashEmbeddingEncoder, MemoryField
from aurora.memory.store import SQLiteMemoryStore
from aurora.runtime.contracts import (
    AtomKind,
    EpisodeContent,
    EvidenceContent,
    MemoryContent,
    MemoryAtom,
    MemoryEdge,
    TranscriptItem,
)
from aurora.runtime.projections import build_memory_brief
import pytest
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


def _session_id(subject_id: str, label: str = "session") -> str:
    return f"{subject_id}-{label}"


def _turn(
    kernel: Any,
    subject_id: str,
    text: str,
    *,
    now_ts: float,
    session_id: str | None = None,
) -> Any:
    return kernel.turn(subject_id, session_id or _session_id(subject_id), text, now_ts=now_ts)


def _finalize(
    kernel: Any,
    subject_id: str,
    *,
    ended_at: float,
    session_id: str | None = None,
) -> Any:
    return kernel.finalize_session(subject_id, session_id or _session_id(subject_id), ended_at=ended_at)


def _stored_atoms(kernel: Any, subject_id: str, *, atom_kind: str | None = None) -> list[MemoryAtom]:
    atoms = list(kernel.store.list_atoms(subject_id))
    if atom_kind is None:
        return atoms
    return [atom for atom in atoms if atom.atom_kind == atom_kind]


def _stored_edges(kernel: Any, subject_id: str) -> list[Any]:
    return list(kernel.store.list_edges(subject_id))


def _activation(kernel: Any, subject_id: str) -> dict[str, float]:
    return dict(kernel.store.list_activation_cache(subject_id))


def _intrinsic_activation(confidence: float, salience: float) -> float:
    retention = confidence * salience
    return retention / (1.0 + retention)


def _add_text_atom(
    kernel: Any,
    *,
    subject_id: str,
    atom_id: str,
    text: str,
    atom_kind: AtomKind = "memory",
    confidence: float = 0.82,
    salience: float = 0.78,
    created_at: float = 0.0,
) -> MemoryAtom:
    atom = MemoryAtom(
        atom_id=atom_id,
        subject_id=subject_id,
        atom_kind=atom_kind,
        content=MemoryContent(text=text),
        confidence=confidence,
        salience=salience,
        created_at=created_at,
    )
    kernel.store.add_atom(atom)
    return atom


def _add_edge(
    kernel: Any,
    *,
    subject_id: str,
    edge_id: str,
    source_atom_id: str,
    target_atom_id: str,
    influence: float,
    confidence: float = 0.95,
    created_at: float = 0.0,
) -> MemoryEdge:
    edge = MemoryEdge(
        edge_id=edge_id,
        subject_id=subject_id,
        source_atom_id=source_atom_id,
        target_atom_id=target_atom_id,
        influence=influence,
        confidence=confidence,
        created_at=created_at,
    )
    kernel.store.add_edge(edge)
    return edge


def test_finalize_session_writes_memory_field_nodes_and_keeps_typed_payloads(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-field"

    turn = _turn(
        kernel,
        subject_id,
        "I work in Hangzhou, I also like jazz, and I am feeling great today.",
        now_ts=1.0,
    )
    state_before = kernel.state(subject_id)
    ingest = _finalize(kernel, subject_id, ended_at=2.0)
    state = kernel.state(subject_id)
    stored_atoms = _stored_atoms(kernel, subject_id)

    assert turn.subject_id == subject_id
    assert turn.session_id == _session_id(subject_id)
    assert turn.response_text.strip() == "ack"
    assert turn.segment_committed is False
    assert state_before.atoms == ()
    assert ingest.created_atom_ids
    assert state.subject_id == subject_id
    assert state.summary.startswith("[CURRENT_MEMORY_FIELD]")
    assert state.atoms
    assert any("Hangzhou" in atom.text for atom in state.atoms)
    assert any("jazz" in atom.text for atom in state.atoms)
    assert any(isinstance(atom.content, EvidenceContent) for atom in stored_atoms)
    assert any(isinstance(atom.content, MemoryContent) for atom in stored_atoms if atom.atom_kind == "memory")
    assert any(isinstance(atom.content, EpisodeContent) for atom in stored_atoms if atom.atom_kind == "episode")


def test_axiom_state_exposes_field_projection_not_truth_schema(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-field-view"

    state = kernel.state(subject_id)
    empty_state_json = _json_text(state)

    assert state.summary == "[CURRENT_MEMORY_FIELD]\n- none"
    assert state.atoms == ()
    assert state.edges == ()
    assert "semantic_self_model" not in empty_state_json
    assert "semantic_world_model" not in empty_state_json
    assert "active_cognition" not in empty_state_json


def test_evidence_only_subject_bootstraps_global_without_topic(kernel_factory: KernelFactory) -> None:
    compiler_calls = 0

    def structured(messages: list[dict[str, str]]) -> str:
        nonlocal compiler_calls
        compiler_calls += 1
        return json.dumps({"atoms": [], "edges": []}, ensure_ascii=False)

    kernel = cast(
        Any,
        kernel_factory(
            llm=QueueLLM(
                "ack",
                repeat_last=True,
                structured=structured,
            )
        ),
    )
    subject_id = "subject-evidence-only"

    state = kernel.state(subject_id)

    assert state.summary == "[CURRENT_MEMORY_FIELD]\n- none"
    assert kernel.store.list_experts(subject_id, expert_kind="global")
    assert kernel.store.list_experts(subject_id, expert_kind="topic") == ()

    turn = _turn(kernel, subject_id, "Just checking in.", now_ts=1.0)
    ingest = _finalize(kernel, subject_id, ended_at=2.0)

    assert turn.response_text == "ack"
    assert compiler_calls == 1
    assert ingest.created_atom_ids == ()
    assert kernel.store.list_experts(subject_id, expert_kind="topic") == ()


def test_bootstrap_failure_rolls_back_partial_subject_state(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    store = SQLiteMemoryStore(str(tmp_path / "aurora_bootstrap_rollback.db"))
    subject_id = "subject-bootstrap-rollback"
    atom = MemoryAtom(
        atom_id="atom-bootstrap-rollback",
        subject_id=subject_id,
        atom_kind="memory",
        content=MemoryContent(text="bootstrap memory"),
        confidence=0.9,
        salience=0.85,
        created_at=1.0,
    )
    store.add_atom(atom)

    def raising_refresh_global(self: MemoryField, subject_id: str, *, now_ts: float) -> None:
        raise RuntimeError("bootstrap failed")

    monkeypatch.setattr(MemoryField, "_refresh_global", raising_refresh_global)
    with pytest.raises(RuntimeError, match="bootstrap failed"):
        MemoryField(store, HashEmbeddingEncoder())

    assert store.list_experts(subject_id) == ()
    assert store.primary_expert_id(subject_id, atom.atom_id) is None
    store.close()


def test_response_reads_memory_brief_before_generation(kernel_factory: KernelFactory) -> None:
    llm = QueueLLM(
        "ack",
        lambda messages: _assert_memory_brief_visible(messages),
        repeat_last=False,
        structured=scripted_memory_llm,
    )
    kernel = cast(Any, kernel_factory(llm=llm))
    subject_id = "subject-pre-response-field"

    _turn(kernel, subject_id, "I now live in Hangzhou.", now_ts=1.0, session_id=_session_id(subject_id, "a"))
    _finalize(kernel, subject_id, session_id=_session_id(subject_id, "a"), ended_at=2.0)
    _turn(kernel, subject_id, "Where do I live now?", now_ts=3.0, session_id=_session_id(subject_id, "b"))


def test_response_reads_open_session_transcript_before_finalize(kernel_factory: KernelFactory) -> None:
    llm = QueueLLM(
        "ack",
        lambda messages: _assert_session_transcript_visible(messages),
        repeat_last=False,
        structured=scripted_memory_llm,
    )
    kernel = cast(Any, kernel_factory(llm=llm))
    subject_id = "subject-open-transcript"
    session_id = _session_id(subject_id)

    _turn(kernel, subject_id, "I live in Hangzhou.", now_ts=1.0, session_id=session_id)
    _turn(kernel, subject_id, "What city am I in right now?", now_ts=2.0, session_id=session_id)
    assert kernel.state(subject_id).atoms == ()


def _assert_memory_brief_visible(messages: list[dict[str, str]]) -> str:
    memory_brief = messages[0]["content"]
    assert "[MEMORY_BRIEF]" in memory_brief
    assert "current_mainline:" in memory_brief
    assert "query_relevant:" in memory_brief
    assert "recent_changes:" in memory_brief
    assert "active_tensions:" in memory_brief
    assert "ongoing_commitments:" in memory_brief
    assert "Hangzhou" in memory_brief
    assert "semantic_self_model" not in memory_brief
    return "ack"


def _assert_session_transcript_visible(messages: list[dict[str, str]]) -> str:
    system_message = messages[0]["content"]
    assert "[ACTIVE_SESSION_TRANSCRIPT]" in system_message
    assert "- user: I live in Hangzhou." in system_message
    assert "- assistant: ack" in system_message
    assert "- user: What city am I in right now?" in system_message
    return "ack"


def test_conflict_write_shifts_activation_without_mutating_old_nodes(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-conflict"
    session_id = _session_id(subject_id)

    _turn(kernel, subject_id, "I live in Shanghai.", now_ts=1.0, session_id=session_id)
    _turn(kernel, subject_id, "Correction, I now live in Hangzhou.", now_ts=2.0, session_id=session_id)
    ingest = _finalize(kernel, subject_id, session_id=session_id, ended_at=3.0)

    state = kernel.state(subject_id)
    stored_memory = _stored_atoms(kernel, subject_id, atom_kind="memory")
    stored_edges = _stored_edges(kernel, subject_id)
    activation = _activation(kernel, subject_id)

    assert len(stored_memory) >= 2
    assert any("Shanghai" in _json_text(atom.content) for atom in stored_memory)
    assert any("Hangzhou" in _json_text(atom.content) for atom in stored_memory)
    assert any(edge.influence < 0.0 for edge in stored_edges)
    assert ingest.created_edge_ids

    shanghai_atom = next(atom for atom in stored_memory if getattr(atom.content, "text", "") == "I now live in Shanghai")
    hangzhou_atom = next(atom for atom in stored_memory if getattr(atom.content, "text", "") == "I now live in Hangzhou")
    assert activation[hangzhou_atom.atom_id] >= activation[shanghai_atom.atom_id]
    assert any("Hangzhou" in atom.text for atom in state.atoms)
    assert all("Shanghai" not in atom.text for atom in state.atoms if atom.atom_kind == "memory")
    assert "status" not in _json_text(stored_memory)

    recalled = kernel.recall(subject_id, "Where do I live?", limit=5)
    assert recalled.atoms
    assert any("Hangzhou" in atom.text for atom in recalled.atoms)


def test_inhibition_changes_activation_without_deleting_nodes(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-inhibition"
    session_id = _session_id(subject_id)

    _turn(kernel, subject_id, "I like jazz.", now_ts=1.0, session_id=session_id)
    _turn(kernel, subject_id, "Please forget that I like jazz.", now_ts=2.0, session_id=session_id)
    _finalize(kernel, subject_id, session_id=session_id, ended_at=3.0)

    state = kernel.state(subject_id)
    stored_atoms = _stored_atoms(kernel, subject_id)
    stored_edges = _stored_edges(kernel, subject_id)
    activation = _activation(kernel, subject_id)

    jazz_atom = next(atom for atom in stored_atoms if atom.atom_kind == "memory" and "jazz" in _json_text(atom.content))
    assert any(atom.atom_kind == "inhibition" for atom in stored_atoms)
    assert any(atom.atom_kind == "evidence" and "I like jazz" in _json_text(atom.content) for atom in stored_atoms)
    assert any(atom.atom_kind == "evidence" and "Please forget that I like jazz" in _json_text(atom.content) for atom in stored_atoms)
    assert any(edge.target_atom_id == jazz_atom.atom_id and edge.influence < 0.0 for edge in stored_edges)
    assert activation[jazz_atom.atom_id] <= 0.05
    assert all(
        not (atom.atom_kind == "memory" and "jazz" in atom.text)
        for atom in state.atoms
    )

    recalled = kernel.recall(subject_id, "What music do I like?", limit=5)
    assert all(
        not (atom.atom_kind == "memory" and "jazz" in atom.text)
        for atom in recalled.atoms
    )


def test_session_compiler_drops_invalid_atoms_and_illegal_edges(kernel_factory: KernelFactory) -> None:
    def structured(messages: list[dict[str, str]]) -> str:
        system_text = "\n".join(message["content"] for message in messages if message["role"] == "system")
        if "[AURORA_SESSION_FIELD_COMPILER]" not in system_text:
            return json.dumps({"atoms": [], "edges": []}, ensure_ascii=False)
        return json.dumps(
            {
                "atoms": [
                    {"kind": "memory", "text": "", "confidence": 0.9, "salience": 0.8},
                    {"kind": "memory", "text": "valid memory", "confidence": 0.91, "salience": 0.83},
                    {"kind": "memory", "text": "invalid score", "confidence": 1.3, "salience": 0.7},
                ],
                "edges": [
                    {"source": "new:0", "target": "missing", "influence": 0.8, "confidence": 0.9},
                    {"source": "new:0", "target": "missing", "influence": 0.8, "confidence": 0.9},
                    {"source": "new:0", "target": "new:0", "influence": 0.8, "confidence": 0.9},
                ],
            },
            ensure_ascii=False,
        )

    kernel = cast(
        Any,
        kernel_factory(
            llm=QueueLLM(
                "ack",
                repeat_last=True,
                structured=structured,
            )
        ),
    )
    subject_id = "subject-compiler-boundary"

    turn = _turn(kernel, subject_id, "Write some noisy content into the field.", now_ts=1.0)
    ingest = _finalize(kernel, subject_id, ended_at=2.0)
    stored_atoms = _stored_atoms(kernel, subject_id)
    stored_edges = _stored_edges(kernel, subject_id)
    stored_memory = [atom for atom in stored_atoms if atom.atom_kind == "memory"]

    assert turn.segment_committed is False
    assert ingest.created_edge_ids == ()
    assert any(atom.atom_kind == "evidence" for atom in stored_atoms)
    assert [atom for atom in stored_atoms if atom.atom_kind == "episode"] == []
    assert len(stored_memory) == 1
    assert _json_text(stored_memory[0].content) == _json_text(MemoryContent(text="valid memory"))
    assert stored_edges == []


def test_axiom_compiler_uses_local_field_not_full_atom_history(kernel_factory: KernelFactory) -> None:
    calls: list[dict[str, object]] = []

    def structured(messages: list[dict[str, str]]) -> str:
        system_text = "\n".join(message["content"] for message in messages if message["role"] == "system")
        if "[AURORA_SESSION_FIELD_COMPILER]" not in system_text:
            return json.dumps({"atoms": [], "edges": []}, ensure_ascii=False)
        payload = json.loads(messages[-1]["content"])
        calls.append(payload)
        return json.dumps({"atoms": [], "edges": []}, ensure_ascii=False)

    kernel = cast(
        Any,
        kernel_factory(
            llm=QueueLLM(
                "ack",
                repeat_last=True,
                structured=structured,
            )
        ),
    )
    subject_id = "subject-local-compiler-context"

    _add_text_atom(
        kernel,
        subject_id=subject_id,
        atom_id="memory-home",
        text="I now live in Hangzhou",
        confidence=0.95,
        salience=0.92,
        created_at=1.0,
    )
    _add_text_atom(
        kernel,
        subject_id=subject_id,
        atom_id="memory-far",
        text="I once visited Reykjavik during a winter trip",
        confidence=0.44,
        salience=0.20,
        created_at=2.0,
    )
    kernel.field.evolve(subject_id)

    _turn(kernel, subject_id, "I still live in Hangzhou and work there.", now_ts=3.0)
    _finalize(kernel, subject_id, ended_at=4.0)

    assert calls
    payload = calls[0]
    local_field = payload.get("local_field")
    assert isinstance(local_field, list)
    local_texts = {
        str(item.get("text", "")).strip()
        for item in local_field
        if isinstance(item, dict)
    }
    assert "I now live in Hangzhou" in local_texts
    assert "I once visited Reykjavik during a winter trip" not in local_texts
    stable_field_summary = payload.get("stable_field_summary")
    assert isinstance(stable_field_summary, str)
    assert stable_field_summary.startswith("[STABLE_FIELD_SUMMARY]")


def test_turn_hot_path_avoids_subject_wide_store_scans_after_bootstrap(
    kernel_factory: KernelFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-sparse-hot-path"

    _turn(kernel, subject_id, "I live in Hangzhou.", now_ts=1.0, session_id=_session_id(subject_id, "a"))
    _finalize(kernel, subject_id, session_id=_session_id(subject_id, "a"), ended_at=2.0)

    subject_wide_calls = {"list_atoms": 0, "list_edges": 0}
    original_list_atoms = kernel.store.list_atoms
    original_list_edges = kernel.store.list_edges

    def tracked_list_atoms(*args: Any, **kwargs: Any) -> Any:
        subject_wide_calls["list_atoms"] += 1
        return original_list_atoms(*args, **kwargs)

    def tracked_list_edges(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("atom_ids") is None:
            subject_wide_calls["list_edges"] += 1
        return original_list_edges(*args, **kwargs)

    monkeypatch.setattr(kernel.store, "list_atoms", tracked_list_atoms)
    monkeypatch.setattr(kernel.store, "list_edges", tracked_list_edges)

    _turn(kernel, subject_id, "I work in Hangzhou and still live there.", now_ts=3.0, session_id=_session_id(subject_id, "b"))

    assert subject_wide_calls == {"list_atoms": 0, "list_edges": 0}


def test_recall_reads_only_routed_experts(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-routed-recall"

    _turn(kernel, subject_id, "I live in Hangzhou and work there.", now_ts=1.0, session_id=_session_id(subject_id, "a"))
    _finalize(kernel, subject_id, session_id=_session_id(subject_id, "a"), ended_at=1.5)
    _turn(kernel, subject_id, "I practice jazz trumpet every weekend.", now_ts=2.0, session_id=_session_id(subject_id, "b"))
    _finalize(kernel, subject_id, session_id=_session_id(subject_id, "b"), ended_at=2.5)
    _turn(kernel, subject_id, "I am planning a winter trip to Reykjavik.", now_ts=3.0, session_id=_session_id(subject_id, "c"))
    _finalize(kernel, subject_id, session_id=_session_id(subject_id, "c"), ended_at=3.5)

    expected_route = kernel.field.route(subject_id, "jazz trumpet", touch=False)
    seen_expert_ids: set[str] = set()
    original_list_expert_atoms = kernel.store.list_expert_atoms

    def tracked_list_expert_atoms(*args: Any, **kwargs: Any) -> Any:
        seen_expert_ids.add(cast(str, args[1]))
        return original_list_expert_atoms(*args, **kwargs)

    kernel.store.list_expert_atoms = tracked_list_expert_atoms
    recall = kernel.recall(subject_id, "jazz trumpet", limit=5)

    assert recall.atoms
    assert seen_expert_ids == set(expected_route.expert_ids)
    assert len(seen_expert_ids) <= 1 + 2


def test_topic_expert_splits_after_threshold(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-topic-split"

    _turn(kernel, subject_id, "I work in alpha systems.", now_ts=1.0)
    route = kernel.field.route(subject_id, "alpha systems", now_ts=2.0, touch=False)
    initial_topic_count = len(kernel.store.list_experts(subject_id, expert_kind="topic"))

    created_atoms: list[MemoryAtom] = []
    for index in range(129):
        atom = MemoryAtom(
            atom_id=f"atom-split-{index}",
            subject_id=subject_id,
            atom_kind="memory",
            content=MemoryContent(text=f"alpha systems cluster memory {index}"),
            confidence=0.8,
            salience=0.75,
            created_at=10.0 + index,
        )
        kernel.store.add_atom(atom)
        created_atoms.append(atom)

    kernel.field.integrate(subject_id, route, tuple(created_atoms), (), now_ts=200.0)

    topic_experts = kernel.store.list_experts(subject_id, expert_kind="topic")
    assert len(topic_experts) > initial_topic_count

    global_cache = kernel.store.list_expert_activation_cache(subject_id, GLOBAL_EXPERT_ID)
    assert global_cache


def test_empty_compile_does_not_leak_topic_experts(kernel_factory: KernelFactory) -> None:
    compiler_calls = 0

    def structured(messages: list[dict[str, str]]) -> str:
        nonlocal compiler_calls
        system_text = "\n".join(message["content"] for message in messages if message["role"] == "system")
        if "[AURORA_SESSION_FIELD_COMPILER]" not in system_text:
            return json.dumps({"atoms": [], "edges": []}, ensure_ascii=False)
        compiler_calls += 1
        if compiler_calls == 1:
            return scripted_memory_llm(messages)
        return json.dumps({"atoms": [], "edges": []}, ensure_ascii=False)

    kernel = cast(
        Any,
        kernel_factory(
            llm=QueueLLM(
                "ack",
                "ack",
                repeat_last=True,
                structured=structured,
            )
        ),
    )
    subject_id = "subject-empty-compile-no-leak"

    _turn(kernel, subject_id, "I live in Hangzhou.", now_ts=1.0, session_id=_session_id(subject_id, "a"))
    _finalize(kernel, subject_id, session_id=_session_id(subject_id, "a"), ended_at=1.5)
    initial_topic_ids = {
        expert.expert_id
        for expert in kernel.store.list_experts(subject_id, expert_kind="topic")
    }

    _turn(
        kernel,
        subject_id,
        "Quantum orchids keep drifting through my notebooks.",
        now_ts=2.0,
        session_id=_session_id(subject_id, "b"),
    )
    _finalize(kernel, subject_id, session_id=_session_id(subject_id, "b"), ended_at=2.5)

    topic_ids = {
        expert.expert_id
        for expert in kernel.store.list_experts(subject_id, expert_kind="topic")
    }
    assert topic_ids == initial_topic_ids


def test_global_state_recomputes_anchors_across_untouched_topics(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-global-anchor-refresh"

    alpha_route = kernel.field.route(subject_id, "alpha systems", now_ts=1.0, touch=False)
    alpha_atoms = tuple(
        MemoryAtom(
            atom_id=f"atom-alpha-{index}",
            subject_id=subject_id,
            atom_kind="memory",
            content=MemoryContent(text=f"alpha systems mainline {index}"),
            confidence=0.91,
            salience=0.90,
            created_at=1.0 + index,
        )
        for index in range(4)
    )
    for atom in alpha_atoms:
        kernel.store.add_atom(atom)
    kernel.field.integrate(subject_id, alpha_route, alpha_atoms, (), now_ts=10.0)

    beta_route = kernel.field.route(subject_id, "beta archives", now_ts=11.0, touch=False)
    beta_atom = MemoryAtom(
        atom_id="atom-beta-anchor",
        subject_id=subject_id,
        atom_kind="memory",
        content=MemoryContent(text="beta archives continuity anchor"),
        confidence=0.90,
        salience=0.88,
        created_at=11.0,
    )
    kernel.store.add_atom(beta_atom)
    kernel.field.integrate(subject_id, beta_route, (beta_atom,), (), now_ts=11.0)

    _, baseline_summary = kernel.field.build_compile_context(
        subject_id,
        "What still matters now?",
        kernel.field.route(subject_id, "What still matters now?", now_ts=11.5, touch=False),
    )
    assert "beta archives continuity anchor" not in baseline_summary

    suppression_route = kernel.field.route(subject_id, "alpha systems", now_ts=12.0, touch=False)
    inhibition = MemoryAtom(
        atom_id="atom-alpha-inhibition",
        subject_id=subject_id,
        atom_kind="inhibition",
        content=MemoryContent(text="alpha systems are no longer current"),
        confidence=0.99,
        salience=0.98,
        created_at=12.0,
    )
    inhibition_edges = tuple(
        MemoryEdge(
            edge_id=f"edge-alpha-suppress-{index}",
            subject_id=subject_id,
            source_atom_id=inhibition.atom_id,
            target_atom_id=atom.atom_id,
            influence=-0.95,
            confidence=0.99,
            created_at=12.0,
        )
        for index, atom in enumerate(alpha_atoms)
    )
    kernel.store.add_atom(inhibition)
    for edge in inhibition_edges:
        kernel.store.add_edge(edge)
    kernel.field.integrate(subject_id, suppression_route, (inhibition,), inhibition_edges, now_ts=12.0)

    state = kernel.state(subject_id)
    assert any("beta archives continuity anchor" in atom.text for atom in state.atoms)

    _, stable_field_summary = kernel.field.build_compile_context(
        subject_id,
        "What still matters now?",
        kernel.field.route(subject_id, "What still matters now?", now_ts=13.0, touch=False),
    )
    assert "beta archives continuity anchor" in stable_field_summary


def test_axiom_malformed_compiler_output_records_compile_failure_evidence(kernel_factory: KernelFactory) -> None:
    kernel = cast(
        Any,
        kernel_factory(
            llm=QueueLLM(
                "ack",
                repeat_last=True,
                structured=lambda messages: "not-json"
                if "[AURORA_SESSION_FIELD_COMPILER]"
                in "\n".join(message["content"] for message in messages if message["role"] == "system")
                else json.dumps({"atoms": [], "edges": []}, ensure_ascii=False),
            )
        ),
    )
    subject_id = "subject-compile-failure"

    turn = _turn(kernel, subject_id, "The compiler will break on this turn.", now_ts=1.0)
    ingest = _finalize(kernel, subject_id, ended_at=2.0)
    stored_atoms = _stored_atoms(kernel, subject_id)
    compile_failures = [
        atom
        for atom in stored_atoms
        if atom.atom_kind == "evidence"
        and isinstance(atom.content, EvidenceContent)
        and atom.content.event_kind == "compile_failure"
    ]

    assert turn.response_text == "ack"
    assert ingest.created_atom_ids
    assert compile_failures
    assert any("Expecting value" in atom.content.text for atom in compile_failures)


def test_query_cannot_resurrect_suppressed_exact_match(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-suppressed-query"

    jazz = _add_text_atom(
        kernel,
        subject_id=subject_id,
        atom_id="memory-jazz",
        text="I like jazz",
        created_at=1.0,
    )
    inhibition = _add_text_atom(
        kernel,
        subject_id=subject_id,
        atom_id="inhibition-jazz",
        atom_kind="inhibition",
        text="Please forget that I like jazz",
        confidence=0.9,
        salience=0.85,
        created_at=2.0,
    )
    _add_edge(
        kernel,
        subject_id=subject_id,
        edge_id="edge-forget-jazz",
        source_atom_id=inhibition.atom_id,
        target_atom_id=jazz.atom_id,
        influence=-1.0,
        confidence=0.95,
        created_at=2.0,
    )

    activation = kernel.field.evolve(subject_id)
    recalled = kernel.recall(subject_id, "jazz", limit=5)

    assert activation[jazz.atom_id] <= 0.05
    assert all(atom.atom_id != jazz.atom_id for atom in recalled.atoms)
    assert any(atom.atom_id == inhibition.atom_id for atom in recalled.atoms)


def test_axiom_intrinsic_activation_depends_only_on_local_retention(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-intrinsic-field"

    anchor = _add_text_atom(
        kernel,
        subject_id=subject_id,
        atom_id="memory-anchor",
        text="anchor memory",
        confidence=0.91,
        salience=0.73,
        created_at=1.0,
    )

    activation = kernel.field.evolve(subject_id)
    expected = _intrinsic_activation(anchor.confidence, anchor.salience)

    assert abs(activation[anchor.atom_id] - expected) < 1e-6


def test_axiom_resting_field_does_not_self_amplify(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-resting-field"

    source = _add_text_atom(
        kernel,
        subject_id=subject_id,
        atom_id="memory-source",
        text="source memory",
        confidence=0.92,
        salience=0.61,
        created_at=1.0,
    )
    target = _add_text_atom(
        kernel,
        subject_id=subject_id,
        atom_id="memory-target",
        text="target memory",
        confidence=0.77,
        salience=0.69,
        created_at=2.0,
    )
    _add_edge(
        kernel,
        subject_id=subject_id,
        edge_id="edge-resting-source-target",
        source_atom_id=source.atom_id,
        target_atom_id=target.atom_id,
        influence=1.0,
        confidence=0.95,
        created_at=3.0,
    )

    activation = kernel.field.evolve(subject_id)

    assert abs(activation[source.atom_id] - _intrinsic_activation(source.confidence, source.salience)) < 1e-6
    assert abs(activation[target.atom_id] - _intrinsic_activation(target.confidence, target.salience)) < 1e-6


def test_completed_turn_writes_episode_and_future_memory_traces(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory, "I will remind you to sync with the team tomorrow.")
    subject_id = "subject-turn-field"

    _turn(kernel, subject_id, "Remind me to sync with the team tomorrow.", now_ts=1.0)
    _finalize(kernel, subject_id, ended_at=2.0)

    state = kernel.state(subject_id)
    stored_atoms = _stored_atoms(kernel, subject_id)

    assert any(atom.atom_kind == "episode" for atom in stored_atoms)
    assert any(atom.atom_kind == "memory" and "Aurora commitment" in _json_text(atom.content) for atom in stored_atoms)
    assert any("Aurora commitment" in atom.text for atom in state.atoms)


def test_axiom_recall_returns_query_activated_field_projection(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-recall-field"

    session_id = _session_id(subject_id)
    _turn(kernel, subject_id, "I work in Hangzhou and I also like jazz.", now_ts=1.0, session_id=session_id)
    _turn(kernel, subject_id, "After moving to Hangzhou, I am readjusting to my work rhythm.", now_ts=2.0, session_id=session_id)
    _turn(
        kernel,
        subject_id,
        "Lately I have kept weekends for friends and exercise, and life feels ordered again.",
        now_ts=3.0,
        session_id=session_id,
    )
    _finalize(kernel, subject_id, session_id=session_id, ended_at=4.0)

    recalled = kernel.recall(subject_id, "Hangzhou adjust life", limit=6)
    memory_brief = build_memory_brief(kernel.state(subject_id), recalled)

    assert recalled.summary.startswith("[QUERY_MEMORY_FIELD]")
    assert recalled.atoms
    assert any(atom.atom_kind == "memory" for atom in recalled.atoms)
    assert memory_brief.startswith("[MEMORY_BRIEF]")
    assert "current_mainline:" in memory_brief
    assert "query_relevant:" in memory_brief
    assert "recent_changes:" in memory_brief

def test_positive_cycle_query_stays_ranked_without_saturating(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-cycle"

    alpha = _add_text_atom(kernel, subject_id=subject_id, atom_id="memory-a", text="Alpha", created_at=1.0)
    beta = _add_text_atom(kernel, subject_id=subject_id, atom_id="memory-b", text="Beta", created_at=2.0)
    gamma = _add_text_atom(kernel, subject_id=subject_id, atom_id="memory-c", text="Gamma", created_at=3.0)
    _add_edge(
        kernel,
        subject_id=subject_id,
        edge_id="edge-a-b",
        source_atom_id=alpha.atom_id,
        target_atom_id=beta.atom_id,
        influence=1.0,
        created_at=4.0,
    )
    _add_edge(
        kernel,
        subject_id=subject_id,
        edge_id="edge-b-c",
        source_atom_id=beta.atom_id,
        target_atom_id=gamma.atom_id,
        influence=1.0,
        created_at=5.0,
    )
    _add_edge(
        kernel,
        subject_id=subject_id,
        edge_id="edge-c-a",
        source_atom_id=gamma.atom_id,
        target_atom_id=alpha.atom_id,
        influence=1.0,
        created_at=6.0,
    )

    activation = kernel.field.evolve(subject_id)
    recalled = kernel.recall(subject_id, "Alpha", limit=3)

    assert max(activation.values()) < 0.8
    assert recalled.atoms
    assert recalled.atoms[0].atom_id == alpha.atom_id
    assert recalled.atoms[0].activation > recalled.atoms[-1].activation
    assert all(atom.activation < 0.8 for atom in recalled.atoms)


def test_axiom_field_evolution_converges_to_fixed_point(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-competitive-fixed-point"

    source = _add_text_atom(kernel, subject_id=subject_id, atom_id="source", text="steady support", created_at=1.0)
    damp = _add_text_atom(
        kernel,
        subject_id=subject_id,
        atom_id="damp",
        atom_kind="inhibition",
        text="steady inhibition",
        confidence=0.9,
        salience=0.84,
        created_at=2.0,
    )
    target = _add_text_atom(kernel, subject_id=subject_id, atom_id="target", text="target memory", created_at=3.0)
    control = _add_text_atom(kernel, subject_id=subject_id, atom_id="control", text="control memory", created_at=4.0)
    _add_edge(
        kernel,
        subject_id=subject_id,
        edge_id="edge-source-target",
        source_atom_id=source.atom_id,
        target_atom_id=target.atom_id,
        influence=1.0,
        created_at=5.0,
    )
    _add_edge(
        kernel,
        subject_id=subject_id,
        edge_id="edge-damp-target",
        source_atom_id=damp.atom_id,
        target_atom_id=target.atom_id,
        influence=-0.95,
        confidence=0.95,
        created_at=6.0,
    )
    _add_edge(
        kernel,
        subject_id=subject_id,
        edge_id="edge-source-control",
        source_atom_id=source.atom_id,
        target_atom_id=control.atom_id,
        influence=1.0,
        created_at=7.0,
    )

    first = kernel.field.evolve(subject_id)
    second = kernel.field.evolve(subject_id)

    assert first == second
    assert 0.0 <= first[target.atom_id] < 0.05
    assert first[control.atom_id] > first[target.atom_id]
    assert first[control.atom_id] >= first[source.atom_id]


def test_axiom_field_activation_is_bounded(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-bounded"

    session_id = _session_id(subject_id)
    _turn(kernel, subject_id, "I live in Shanghai.", now_ts=1.0, session_id=session_id)
    _turn(kernel, subject_id, "Correction, I now live in Hangzhou.", now_ts=2.0, session_id=session_id)
    _turn(
        kernel,
        subject_id,
        "I want to first go for a walk, then come back and write docs.",
        now_ts=3.0,
        session_id=session_id,
    )
    _turn(kernel, subject_id, "I feel anxious today.", now_ts=4.0, session_id=session_id)
    _finalize(kernel, subject_id, session_id=session_id, ended_at=5.0)

    activation = _activation(kernel, subject_id)

    assert activation
    assert all(0.0 <= value <= 1.0 for value in activation.values())


def test_axiom_recall_is_read_only_against_cached_field(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-read-only-recall"

    _turn(kernel, subject_id, "I live in Hangzhou and I also like jazz.", now_ts=1.0)
    _finalize(kernel, subject_id, ended_at=2.0)
    before = _activation(kernel, subject_id)

    first = kernel.recall(subject_id, "Hangzhou", limit=5)
    after_first = _activation(kernel, subject_id)
    second = kernel.recall(subject_id, "Hangzhou", limit=5)
    after_second = _activation(kernel, subject_id)

    assert before == after_first == after_second
    assert first == second


def test_axiom_repeated_recall_does_not_drift_field_projection(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-recall-sequence"

    session_id = _session_id(subject_id)
    _turn(kernel, subject_id, "I work in Hangzhou and I also like jazz.", now_ts=1.0, session_id=session_id)
    _turn(kernel, subject_id, "Recently I have been adjusting to a new pace of life.", now_ts=2.0, session_id=session_id)
    _finalize(kernel, subject_id, session_id=session_id, ended_at=3.0)
    baseline_state = kernel.state(subject_id)
    baseline_activation = _activation(kernel, subject_id)
    baseline_recall = kernel.recall(subject_id, "Hangzhou", limit=5)

    for query in ("Hangzhou", "life", "jazz", "work", "adjust", "Hangzhou"):
        kernel.recall(subject_id, query, limit=5)

    final_state = kernel.state(subject_id)
    final_activation = _activation(kernel, subject_id)
    final_recall = kernel.recall(subject_id, "Hangzhou", limit=5)

    assert baseline_activation == final_activation
    assert baseline_state == final_state
    assert baseline_recall == final_recall


def test_state_and_recall_ignore_open_segment_until_finalize(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-open-segment-hidden"

    _turn(kernel, subject_id, "I live in Hangzhou.", now_ts=1.0)

    assert kernel.state(subject_id).atoms == ()
    assert kernel.recall(subject_id, "Hangzhou", limit=5).atoms == ()

    _finalize(kernel, subject_id, ended_at=2.0)

    assert any("Hangzhou" in atom.text for atom in kernel.state(subject_id).atoms)
    assert any("Hangzhou" in atom.text for atom in kernel.recall(subject_id, "Hangzhou", limit=5).atoms)


def test_finalize_session_closes_session_and_rejects_repeat_turns(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-session-close"
    session_id = _session_id(subject_id)

    _turn(kernel, subject_id, "I work in Hangzhou.", now_ts=1.0, session_id=session_id)
    _finalize(kernel, subject_id, session_id=session_id, ended_at=2.0)

    with pytest.raises(RuntimeError, match="already finalized"):
        _finalize(kernel, subject_id, session_id=session_id, ended_at=3.0)

    with pytest.raises(RuntimeError, match="already finalized"):
        _turn(kernel, subject_id, "Can I keep talking here?", now_ts=4.0, session_id=session_id)


def test_ingest_transcript_accepts_unpaired_message_order(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-batch-ingest"

    output = kernel.ingest_transcript(
        subject_id,
        _session_id(subject_id),
        (
            TranscriptItem(role="assistant", text="Hello again.", created_at=1.0),
            TranscriptItem(role="user", text="I live in Hangzhou.", created_at=2.0),
            TranscriptItem(role="assistant", text="Noted.", created_at=3.0),
        ),
        ended_at=4.0,
    )

    assert output.created_atom_ids
    assert any("Hangzhou" in atom.text for atom in kernel.state(subject_id).atoms)


def test_axiom_inhibitory_pressure_keeps_suppressed_node_hidden(kernel_factory: KernelFactory) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-neighbor-suppression"

    suppressed = _add_text_atom(
        kernel,
        subject_id=subject_id,
        atom_id="suppressed",
        text="I like jazz",
        created_at=1.0,
    )
    neighbor = _add_text_atom(
        kernel,
        subject_id=subject_id,
        atom_id="neighbor",
        atom_kind="memory",
        text="I have been listening to many kinds of music lately",
        confidence=0.8,
        salience=0.76,
        created_at=2.0,
    )
    inhibition = _add_text_atom(
        kernel,
        subject_id=subject_id,
        atom_id="inhibition",
        atom_kind="inhibition",
        text="Please forget that I like jazz",
        confidence=0.9,
        salience=0.85,
        created_at=3.0,
    )
    _add_edge(
        kernel,
        subject_id=subject_id,
        edge_id="edge-neighbor-suppressed",
        source_atom_id=neighbor.atom_id,
        target_atom_id=suppressed.atom_id,
        influence=0.9,
        confidence=0.9,
        created_at=4.0,
    )
    _add_edge(
        kernel,
        subject_id=subject_id,
        edge_id="edge-inhibition-suppressed",
        source_atom_id=inhibition.atom_id,
        target_atom_id=suppressed.atom_id,
        influence=-1.0,
        confidence=0.95,
        created_at=5.0,
    )

    baseline = kernel.field.evolve(subject_id)
    before = _activation(kernel, subject_id)

    for query in ("music", "what have I been listening to lately", "jazz", "music", "what have I been listening to lately"):
        recalled = kernel.recall(subject_id, query, limit=5)
        assert all(atom.atom_id != suppressed.atom_id for atom in recalled.atoms)

    after = _activation(kernel, subject_id)
    state = kernel.state(subject_id)

    assert baseline[suppressed.atom_id] <= 0.05
    assert before == after
    assert all(atom.atom_id != suppressed.atom_id for atom in state.atoms)
