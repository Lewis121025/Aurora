from __future__ import annotations

from aurora.field_engine import RecallEdge, RecallItem, RecallResult
from aurora.projections import build_memory_brief


def _item(
    atom_id: str,
    text: str,
    *,
    kind: str = "fact",
    score: float = 0.8,
    birth_step: int = 1,
    signature: tuple[str, ...] = (),
) -> RecallItem:
    return RecallItem(
        atom_id=atom_id,
        kind=kind,
        score=score,
        text=text,
        payload={"text": text, "signature": list(signature)},
        reason={},
        birth_step=birth_step,
        source="dialogue",
    )


def _edge(src: str, dst: str, kind: str, weight: float = 0.8, confidence: float = 0.9) -> RecallEdge:
    return RecallEdge(src=src, dst=dst, kind=kind, weight=weight, confidence=confidence)


def _result(
    cue: str,
    *items: RecallItem,
    edges: list[RecallEdge] | None = None,
) -> RecallResult:
    return RecallResult(cue=cue, items=list(items), edges=list(edges or []), trace={})


def _sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        if not line or line == "[MEMORY_BRIEF]":
            continue
        if line.endswith(":"):
            current = line[:-1]
            sections[current] = []
            continue
        if current is not None and line.startswith("- "):
            sections[current].append(line[2:])
    return sections


def test_memory_brief_keeps_commitments_out_of_mainline() -> None:
    state = _result(
        "state",
        _item("fact-home", "I live in Hangzhou", score=0.95, signature=("self", "live")),
        _item("fact-job", "I work in Hangzhou", score=0.92, signature=("self", "work")),
        _item("fact-commitment", "Aurora commitment: remind you tomorrow", score=0.99),
    )

    sections = _sections(build_memory_brief(state))

    assert sections["current_mainline"] == ["I live in Hangzhou", "I work in Hangzhou"]
    assert sections["ongoing_commitments"] == ["Aurora commitment: remind you tomorrow"]


def test_memory_brief_recent_changes_prefers_newer_items() -> None:
    state = _result(
        "state",
        _item("fact-old", "I lived in Shanghai", birth_step=1, signature=("self", "live")),
        _item("fact-mid", "I live in Hangzhou", birth_step=2, signature=("self", "live")),
    )
    recall = _result(
        "query",
        _item("abstract-now", "[ABSTRACT] live / hangzhou", kind="abstract", birth_step=3),
    )

    sections = _sections(build_memory_brief(state, recall))

    assert sections["recent_changes"] == [
        "[ABSTRACT] live / hangzhou",
        "I live in Hangzhou",
        "I lived in Shanghai",
    ]


def test_memory_brief_renders_negative_edges_as_tensions() -> None:
    state = _result(
        "state",
        _item("fact-home-new", "I live in Hangzhou", signature=("self", "live")),
        _item("fact-home-old", "I live in Shanghai", signature=("self", "live")),
        edges=[_edge("fact-home-new", "fact-home-old", "suppresses"), _edge("fact-home-old", "fact-home-new", "contradicts")],
    )

    sections = _sections(build_memory_brief(state))

    assert sections["active_tensions"] == [
        "I live in Shanghai is suppressed by I live in Hangzhou",
        "I live in Shanghai conflicts with I live in Hangzhou",
    ]


def test_memory_brief_dedupes_same_signature_in_query_relevant() -> None:
    recall = _result(
        "query",
        _item("fact-home-new", "I live in Hangzhou", score=0.95, signature=("self", "live")),
        _item("fact-home-old", "I live in Shanghai", score=0.80, signature=("self", "live")),
        _item("fact-like", "I like tea", score=0.78, signature=("self", "like")),
    )

    sections = _sections(build_memory_brief(_result("state"), recall))

    assert sections["query_relevant"] == ["I live in Hangzhou", "I like tea"]
