from __future__ import annotations

import json
from typing import Any, cast

from aurora.runtime.contracts import EvidenceContent, MemoryAtom
from aurora.runtime.projections import build_memory_brief
from tests.conftest import KernelFactory, QueueLLM, scripted_memory_llm


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


def _activation(kernel: Any, subject_id: str) -> dict[str, float]:
    return dict(kernel.store.list_activation_cache(subject_id))


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _sections(memory_brief: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in memory_brief.splitlines():
        if not line or line == "[MEMORY_BRIEF]":
            continue
        if line.endswith(":"):
            current = line[:-1]
            sections[current] = []
            continue
        if current is not None and line.startswith("- "):
            sections[current].append(line[2:])
    return sections


def test_release_scenario_correction_replaces_current_dominance_without_mutating_old_atoms(
    kernel_factory: KernelFactory,
) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-release-correction"
    session_id = _session_id(subject_id)

    _turn(kernel, subject_id, "I live in Shanghai.", now_ts=1.0, session_id=session_id)
    _turn(kernel, subject_id, "Correction, I now live in Hangzhou.", now_ts=2.0, session_id=session_id)
    _finalize(kernel, subject_id, session_id=session_id, ended_at=3.0)

    stored_memory = _stored_atoms(kernel, subject_id, atom_kind="memory")
    activation = _activation(kernel, subject_id)
    state = kernel.state(subject_id)
    recall = kernel.recall(subject_id, "Where do I live now?", limit=5)
    brief = build_memory_brief(state, recall)

    shanghai_atom = next(atom for atom in stored_memory if getattr(atom.content, "text", "") == "I now live in Shanghai")
    hangzhou_atom = next(atom for atom in stored_memory if getattr(atom.content, "text", "") == "I now live in Hangzhou")

    assert any("Shanghai" in _json_text(atom.content) for atom in stored_memory)
    assert any("Hangzhou" in _json_text(atom.content) for atom in stored_memory)
    assert activation[hangzhou_atom.atom_id] >= activation[shanghai_atom.atom_id]
    assert any("Hangzhou" in atom.text for atom in state.atoms)
    assert all("Shanghai" not in atom.text for atom in state.atoms if atom.atom_kind == "memory")
    assert any("Hangzhou" in atom.text for atom in recall.atoms)
    assert "Hangzhou" in brief


def test_release_scenario_inhibition_hides_memory_without_deleting_evidence(
    kernel_factory: KernelFactory,
) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-release-inhibition"
    session_id = _session_id(subject_id)

    _turn(kernel, subject_id, "I like jazz.", now_ts=1.0, session_id=session_id)
    _turn(kernel, subject_id, "Please forget that I like jazz.", now_ts=2.0, session_id=session_id)
    _finalize(kernel, subject_id, session_id=session_id, ended_at=3.0)

    stored_atoms = _stored_atoms(kernel, subject_id)
    state = kernel.state(subject_id)
    recall = kernel.recall(subject_id, "What music do I like?", limit=5)
    brief = build_memory_brief(state, recall)
    sections = _sections(brief)

    assert any(
        atom.atom_kind == "evidence"
        and isinstance(atom.content, EvidenceContent)
        and atom.content.text == "I like jazz."
        for atom in stored_atoms
    )
    assert any(
        atom.atom_kind == "evidence"
        and isinstance(atom.content, EvidenceContent)
        and atom.content.text == "Please forget that I like jazz."
        for atom in stored_atoms
    )
    assert all("jazz" not in atom.text.lower() for atom in state.atoms if atom.atom_kind == "memory")
    assert all("jazz" not in atom.text.lower() for atom in recall.atoms if atom.atom_kind == "memory")
    expected_episode = "user: I like jazz. | assistant: ack | user: Please forget that I like jazz. | assistant: ack"
    assert sections["current_mainline"] == [expected_episode]
    assert sections["query_relevant"] == [expected_episode]


def test_release_scenario_repeated_recall_does_not_drift_cached_field_state(
    kernel_factory: KernelFactory,
) -> None:
    kernel = _kernel(kernel_factory)
    subject_id = "subject-release-read-only"
    session_id = _session_id(subject_id)

    _turn(kernel, subject_id, "I work in Hangzhou and I also like jazz.", now_ts=1.0, session_id=session_id)
    _turn(kernel, subject_id, "Recently I have been adjusting to a new pace of life.", now_ts=2.0, session_id=session_id)
    _finalize(kernel, subject_id, session_id=session_id, ended_at=3.0)

    baseline_activation = _activation(kernel, subject_id)
    baseline_state = kernel.state(subject_id)
    baseline_recall = kernel.recall(subject_id, "Hangzhou", limit=5)
    baseline_brief = build_memory_brief(baseline_state, baseline_recall)

    for query in ("Hangzhou", "life", "jazz", "work", "adjust", "Hangzhou"):
        kernel.recall(subject_id, query, limit=5)

    assert baseline_activation == _activation(kernel, subject_id)
    assert baseline_state == kernel.state(subject_id)
    assert baseline_recall == kernel.recall(subject_id, "Hangzhou", limit=5)
    assert baseline_brief == build_memory_brief(kernel.state(subject_id), kernel.recall(subject_id, "Hangzhou", limit=5))


def test_release_scenario_commitment_remains_visible_across_subsequent_turns(
    kernel_factory: KernelFactory,
) -> None:
    kernel = _kernel(kernel_factory, "I will remind you to sync with the team tomorrow.")
    subject_id = "subject-release-commitment"
    session_id = _session_id(subject_id)

    _turn(kernel, subject_id, "Remind me to sync with the team tomorrow.", now_ts=1.0, session_id=session_id)
    _turn(kernel, subject_id, "I also work in Hangzhou.", now_ts=2.0, session_id=session_id)
    _finalize(kernel, subject_id, session_id=session_id, ended_at=3.0)

    state = kernel.state(subject_id)
    recall = kernel.recall(subject_id, "What do you still owe me?", limit=5)
    brief = build_memory_brief(state, recall)

    assert any("Aurora commitment:" in atom.text for atom in state.atoms)
    assert "ongoing_commitments:" in brief
    assert "Aurora commitment:" in brief


def test_release_scenario_malformed_compiler_output_records_explicit_failure_evidence(
    kernel_factory: KernelFactory,
) -> None:
    compiler_calls = 0

    def structured(messages: list[dict[str, str]]) -> str:
        nonlocal compiler_calls
        system_text = "\n".join(message["content"] for message in messages if message["role"] == "system")
        if "[AURORA_SESSION_FIELD_COMPILER]" not in system_text:
            return json.dumps({"atoms": [], "edges": []}, ensure_ascii=False)
        compiler_calls += 1
        if compiler_calls == 1:
            return scripted_memory_llm(messages)
        return "not-json"

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
    subject_id = "subject-release-compile-failure"

    _turn(kernel, subject_id, "I live in Hangzhou.", now_ts=1.0, session_id=_session_id(subject_id, "a"))
    _finalize(kernel, subject_id, session_id=_session_id(subject_id, "a"), ended_at=1.5)
    initial_topic_count = len(kernel.store.list_experts(subject_id, expert_kind="topic"))
    _turn(
        kernel,
        subject_id,
        "This turn should force a compiler failure.",
        now_ts=2.0,
        session_id=_session_id(subject_id, "b"),
    )
    _finalize(kernel, subject_id, session_id=_session_id(subject_id, "b"), ended_at=2.5)

    compile_failures = [
        atom
        for atom in _stored_atoms(kernel, subject_id)
        if atom.atom_kind == "evidence"
        and isinstance(atom.content, EvidenceContent)
        and atom.content.event_kind == "compile_failure"
    ]

    assert compile_failures
    assert any("Expecting value" in atom.content.text for atom in compile_failures)
    assert len(kernel.store.list_experts(subject_id, expert_kind="topic")) == initial_topic_count
