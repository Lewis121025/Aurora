from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

from aurora.relation.decision import RelationDecisionContext
from aurora.runtime.contracts import TraceChannel


@lru_cache(maxsize=1)
def _touch_lexicon() -> dict[str, dict[str, float]]:
    with (
        resources.files("aurora.being")
        .joinpath("touch_lexicon.json")
        .open("r", encoding="utf-8") as handle
    ):
        data = json.load(handle)
    return {
        str(category): {str(token): float(weight) for token, weight in values.items()}
        for category, values in data.items()
    }


def match_touch_scores(text: str) -> dict[str, float]:
    lowered = text.lower()
    scores: dict[str, float] = {}
    for category, tokens in _touch_lexicon().items():
        matched = [weight for token, weight in tokens.items() if token in text or token in lowered]
        if matched:
            scores[category] = max(matched)
    return scores


LEXICAL_BASE = 0.35
LEXICAL_HISTORY_SCALE = 0.65
HISTORY_INFER_INTENSITY = 0.3

CHANNEL_TO_CATEGORY: dict[TraceChannel, str] = {
    TraceChannel.WARMTH: "warmth",
    TraceChannel.RECOGNITION: "insight",
    TraceChannel.HURT: "hurt",
    TraceChannel.BOUNDARY: "boundary",
    TraceChannel.CURIOSITY: "curiosity",
    TraceChannel.REPAIR: "repair",
    TraceChannel.DISTANCE: "distance",
}


def graph_mediated_touch_scores(
    lexical_scores: dict[str, float],
    recalled_channels: tuple[TraceChannel, ...],
    relation_context: RelationDecisionContext,
) -> dict[str, float]:
    channel_set = set(recalled_channels)
    has_history = bool(channel_set) or (
        relation_context.boundary_events
        + relation_context.repair_events
        + relation_context.resonance_events
        + relation_context.thread_count
        + relation_context.knot_count
        > 0
    )
    effective_scores = lexical_scores
    if not lexical_scores and has_history:
        effective_scores = _infer_from_history(channel_set, relation_context)
    scores: dict[str, float] = {}
    for category, lexical in effective_scores.items():
        support = _support_strength(category, channel_set, relation_context, has_history)
        scores[category] = round(min(1.0, lexical * (LEXICAL_BASE + LEXICAL_HISTORY_SCALE * support)), 4)
    return scores


def _infer_from_history(
    channels: set[TraceChannel],
    relation: RelationDecisionContext,
) -> dict[str, float]:
    inferred: dict[str, float] = {}
    for channel in channels:
        category = CHANNEL_TO_CATEGORY.get(channel)
        if category is not None:
            inferred[category] = max(inferred.get(category, 0.0), HISTORY_INFER_INTENSITY)
    if relation.boundary_events > 0 and "boundary" not in inferred:
        inferred["boundary"] = HISTORY_INFER_INTENSITY * 0.6
    if relation.repair_events > 0 and "repair" not in inferred:
        inferred["repair"] = HISTORY_INFER_INTENSITY * 0.5
    return inferred


def question_touch_signal(text: str) -> float:
    return 0.6 if any(token in text for token in ("?", "？")) else 0.0


def _support_strength(
    category: str,
    channels: set[TraceChannel],
    relation: RelationDecisionContext,
    has_history: bool,
) -> float:
    support = 0.2 if not has_history else 0.0
    if category == "warmth":
        if TraceChannel.WARMTH in channels or TraceChannel.RECOGNITION in channels:
            support += 0.45
        if relation.resonance_events >= relation.boundary_events:
            support += 0.2
        if relation.thread_count > 0:
            support += 0.15
    elif category == "insight":
        if TraceChannel.RECOGNITION in channels or TraceChannel.COHERENCE in channels:
            support += 0.35
        if relation.thread_count > 0:
            support += 0.15
        if relation.resonance_events > 0:
            support += 0.1
    elif category == "hurt":
        if TraceChannel.HURT in channels or TraceChannel.BOUNDARY in channels:
            support += 0.4
        if relation.knot_count > 0:
            support += 0.2
        if relation.boundary_events > 0:
            support += 0.15
    elif category == "boundary":
        if TraceChannel.BOUNDARY in channels:
            support += 0.45
        if relation.boundary_events > 0:
            support += 0.25
        if relation.knot_count > 0:
            support += 0.1
    elif category == "curiosity":
        if TraceChannel.CURIOSITY in channels or TraceChannel.WONDER in channels:
            support += 0.35
        if TraceChannel.COHERENCE in channels:
            support += 0.15
        if relation.boundary_events == 0 or relation.repair_events >= relation.boundary_events:
            support += 0.1
    elif category == "repair":
        if TraceChannel.REPAIR in channels:
            support += 0.4
        if relation.repair_events > 0:
            support += 0.2
        if relation.resonance_events > 0:
            support += 0.1
    elif category == "distance":
        if TraceChannel.DISTANCE in channels:
            support += 0.4
        if relation.boundary_events > relation.resonance_events:
            support += 0.2
        if relation.knot_count > 0:
            support += 0.15
    return min(1.0, support)
