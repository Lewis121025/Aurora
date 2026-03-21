"""Memory-brief projections for response generation."""

from __future__ import annotations

from collections.abc import Iterable

from aurora.field_engine import RecallEdge, RecallItem, RecallResult

_MAINLINE_LIMIT = 4
_QUERY_LIMIT = 4
_RECENT_LIMIT = 3
_TENSION_LIMIT = 3
_COMMITMENT_LIMIT = 3
_KIND_PRIORITY = {"fact": 0, "abstract": 1, "anchor": 2}


def build_memory_brief(
    state: RecallResult,
    recall: RecallResult | None = None,
    *,
    heading: str = "[MEMORY_BRIEF]",
) -> str:
    recall_items = [] if recall is None else recall.items
    recall_edges = [] if recall is None else recall.edges
    current_mainline = _current_mainline(state.items)
    query_relevant = _query_relevant(recall_items)
    recent_changes = _recent_changes(state.items, recall_items)
    active_tensions = _active_tensions(state.items, recall_items, state.edges, recall_edges)
    ongoing_commitments = _ongoing_commitments(state.items, recall_items)
    lines = [heading]
    _append_items(lines, "current_mainline", current_mainline)
    _append_items(lines, "query_relevant", query_relevant)
    _append_items(lines, "recent_changes", recent_changes)
    _append_text(lines, "active_tensions", active_tensions)
    _append_items(lines, "ongoing_commitments", ongoing_commitments)
    return "\n".join(lines)


def _current_mainline(items: list[RecallItem]) -> list[RecallItem]:
    ranked = sorted(
        (item for item in _dedupe_items(items) if item.kind != "anchor" and not _is_commitment(item.text)),
        key=lambda item: (_kind_rank(item), -item.score, -item.birth_step, item.atom_id),
    )
    return _limit_unique_signatures(ranked, _MAINLINE_LIMIT)


def _query_relevant(items: list[RecallItem]) -> list[RecallItem]:
    ranked = sorted(
        (item for item in _dedupe_items(items) if not _is_commitment(item.text)),
        key=lambda item: (_kind_rank(item), -item.score, -item.birth_step, item.atom_id),
    )
    return _limit_unique_signatures(ranked, _QUERY_LIMIT)


def _recent_changes(state_items: list[RecallItem], recall_items: list[RecallItem]) -> list[RecallItem]:
    ranked = sorted(
        _dedupe_items([*recall_items, *state_items]),
        key=lambda item: (-item.birth_step, _kind_rank(item), -item.score, item.atom_id),
    )
    return ranked[:_RECENT_LIMIT]


def _active_tensions(
    state_items: list[RecallItem],
    recall_items: list[RecallItem],
    state_edges: list[RecallEdge],
    recall_edges: list[RecallEdge],
) -> list[str]:
    items_by_id = {item.atom_id: item for item in _dedupe_items([*recall_items, *state_items])}
    lines: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for edge in sorted(
        [*recall_edges, *state_edges],
        key=lambda item: (item.kind not in {"suppresses", "contradicts"}, -(item.weight * item.confidence), item.src, item.dst),
    ):
        if edge.kind not in {"suppresses", "contradicts"}:
            continue
        key = (edge.src, edge.dst, edge.kind)
        if key in seen:
            continue
        source = items_by_id.get(edge.src)
        target = items_by_id.get(edge.dst)
        if source is None or target is None:
            continue
        seen.add(key)
        if edge.kind == "suppresses":
            lines.append(f"{target.text} is suppressed by {source.text}")
        else:
            lines.append(f"{source.text} conflicts with {target.text}")
        if len(lines) >= _TENSION_LIMIT:
            break
    return lines


def _ongoing_commitments(state_items: list[RecallItem], recall_items: list[RecallItem]) -> list[RecallItem]:
    ranked = sorted(
        (item for item in _dedupe_items([*recall_items, *state_items]) if _is_commitment(item.text)),
        key=lambda item: (_kind_rank(item), -item.score, -item.birth_step, item.atom_id),
    )
    return _limit_unique_signatures(ranked, _COMMITMENT_LIMIT)


def _dedupe_items(items: Iterable[RecallItem]) -> list[RecallItem]:
    seen: set[str] = set()
    ordered: list[RecallItem] = []
    for item in items:
        if item.atom_id in seen:
            continue
        seen.add(item.atom_id)
        ordered.append(item)
    return ordered


def _limit_unique_signatures(items: list[RecallItem], limit: int) -> list[RecallItem]:
    winners: list[RecallItem] = []
    seen_signatures: set[tuple[str, ...]] = set()
    for item in items:
        signature = tuple(str(token) for token in item.payload.get("signature", ()))
        if signature and signature in seen_signatures:
            continue
        if signature:
            seen_signatures.add(signature)
        winners.append(item)
        if len(winners) >= limit:
            break
    return winners


def _append_items(lines: list[str], heading: str, items: list[RecallItem]) -> None:
    lines.append(f"{heading}:")
    if not items:
        lines.append("- none")
        return
    for item in items:
        lines.append(f"- {item.text}")


def _append_text(lines: list[str], heading: str, items: list[str]) -> None:
    lines.append(f"{heading}:")
    if not items:
        lines.append("- none")
        return
    for item in items:
        lines.append(f"- {item}")


def _kind_rank(item: RecallItem) -> int:
    return _KIND_PRIORITY.get(item.kind, 1)


def _is_commitment(text: str) -> bool:
    return text.startswith("Aurora commitment:")
