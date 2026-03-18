from __future__ import annotations

from typing import Any, cast

from aurora.runtime.contracts import ActivatedAtom, ActivatedEdge, RecallResult, SubjectMemoryState
from aurora.runtime.projections import build_memory_brief


def _atom(
    atom_id: str,
    text: str,
    *,
    atom_kind: str = "memory",
    activation: float = 0.8,
    confidence: float = 0.85,
    salience: float = 0.8,
    created_at: float = 0.0,
) -> ActivatedAtom:
    return ActivatedAtom(
        atom_id=atom_id,
        atom_kind=cast(Any, atom_kind),
        text=text,
        activation=activation,
        confidence=confidence,
        salience=salience,
        created_at=created_at,
    )


def _edge(
    source_atom_id: str,
    target_atom_id: str,
    *,
    influence: float,
    confidence: float = 0.9,
) -> ActivatedEdge:
    return ActivatedEdge(
        source_atom_id=source_atom_id,
        target_atom_id=target_atom_id,
        influence=influence,
        confidence=confidence,
    )


def _state(
    *atoms: ActivatedAtom,
    edges: tuple[ActivatedEdge, ...] = (),
    subject_id: str = "subject-brief",
) -> SubjectMemoryState:
    return SubjectMemoryState(
        subject_id=subject_id,
        summary="[CURRENT_MEMORY_FIELD]",
        atoms=atoms,
        edges=edges,
    )


def _recall(
    *atoms: ActivatedAtom,
    edges: tuple[ActivatedEdge, ...] = (),
    subject_id: str = "subject-brief",
    query: str = "杭州",
) -> RecallResult:
    return RecallResult(
        subject_id=subject_id,
        query=query,
        summary="[QUERY_MEMORY_FIELD]",
        atoms=atoms,
        edges=edges,
    )


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


def test_memory_brief_current_mainline_excludes_commitments() -> None:
    state = _state(
        _atom("memory-home", "我现在住在杭州", activation=0.93),
        _atom("memory-job", "我在杭州工作", activation=0.88),
        _atom("memory-commitment", "Aurora承诺：明天提醒同步团队", activation=0.95),
    )

    sections = _sections(build_memory_brief(state))

    assert sections["current_mainline"] == ["我现在住在杭州", "我在杭州工作"]
    assert sections["ongoing_commitments"] == ["Aurora承诺：明天提醒同步团队"]


def test_memory_brief_current_mainline_falls_back_to_episode_when_no_plain_memory() -> None:
    state = _state(
        _atom("episode-latest", "在杭州重新适应新的生活节奏", atom_kind="episode", created_at=2.0),
        _atom("memory-commitment", "Aurora承诺：明天提醒同步团队", activation=0.95),
    )

    sections = _sections(build_memory_brief(state))

    assert sections["current_mainline"] == ["在杭州重新适应新的生活节奏"]


def test_memory_brief_query_relevant_excludes_inhibition_and_keeps_overlap_with_mainline() -> None:
    state = _state(_atom("memory-home", "我现在住在杭州", activation=0.93))
    recall = _recall(
        _atom("memory-home", "我现在住在杭州", activation=0.97),
        _atom("inhibition-home", "请忘掉我住在上海", atom_kind="inhibition", activation=0.91),
        query="我现在住在哪里？",
    )

    sections = _sections(build_memory_brief(state, recall))

    assert sections["current_mainline"] == ["我现在住在杭州"]
    assert sections["query_relevant"] == ["我现在住在杭州"]


def test_memory_brief_recent_changes_uses_newest_episodes_across_state_and_recall() -> None:
    state = _state(
        _atom("episode-old", "刚搬到杭州", atom_kind="episode", created_at=1.0),
        _atom("episode-mid", "开始适应新的工作节奏", atom_kind="episode", created_at=2.0),
    )
    recall = _recall(
        _atom("episode-new", "最近我在重新适应新的生活节奏", atom_kind="episode", created_at=3.0),
        _atom("episode-mid", "开始适应新的工作节奏", atom_kind="episode", created_at=2.0),
    )

    sections = _sections(build_memory_brief(state, recall))

    assert sections["recent_changes"] == [
        "最近我在重新适应新的生活节奏",
        "开始适应新的工作节奏",
        "刚搬到杭州",
    ]


def test_memory_brief_active_tensions_renders_inhibition_and_negative_coupling() -> None:
    home = _atom("memory-home", "我现在住在杭州")
    social = _atom("memory-social", "我想多社交")
    solitude = _atom("memory-solitude", "我也想保留更多独处时间")
    inhibition = _atom("inhibition-home", "请忘掉我住在上海", atom_kind="inhibition")
    state = _state(
        home,
        social,
        solitude,
        inhibition,
        edges=(
            _edge("memory-social", "memory-solitude", influence=-0.72, confidence=0.91),
            _edge("inhibition-home", "memory-home", influence=-0.95, confidence=0.96),
        ),
    )
    recall = _recall(
        home,
        social,
        solitude,
        inhibition,
        edges=(
            _edge("memory-social", "memory-solitude", influence=-0.72, confidence=0.91),
            _edge("inhibition-home", "memory-home", influence=-0.95, confidence=0.96),
        ),
    )

    sections = _sections(build_memory_brief(state, recall))

    assert sections["active_tensions"] == [
        "我现在住在杭州 受到 请忘掉我住在上海 的抑制",
        "我想多社交 与 我也想保留更多独处时间 存在张力",
    ]
