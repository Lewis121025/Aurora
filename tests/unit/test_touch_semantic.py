from __future__ import annotations

from aurora.being.touch import graph_mediated_touch_scores, match_touch_scores
from aurora.relation.decision import RelationDecisionContext
from aurora.runtime.contracts import TraceChannel


def _ctx(
    *,
    boundary_events: int = 0,
    repair_events: int = 0,
    resonance_events: int = 0,
    thread_count: int = 0,
    knot_count: int = 0,
) -> RelationDecisionContext:
    return RelationDecisionContext(
        boundary_events=boundary_events,
        repair_events=repair_events,
        resonance_events=resonance_events,
        thread_count=thread_count,
        knot_count=knot_count,
    )


def test_empty_text_produces_no_lexical_scores() -> None:
    assert match_touch_scores("") == {}


def test_history_only_touch_when_no_keywords() -> None:
    scores = graph_mediated_touch_scores(
        lexical_scores={},
        recalled_channels=(TraceChannel.WARMTH, TraceChannel.RECOGNITION),
        relation_context=_ctx(resonance_events=2, thread_count=1),
    )
    assert len(scores) > 0
    assert "warmth" in scores or "insight" in scores


def test_history_only_touch_empty_when_no_history() -> None:
    scores = graph_mediated_touch_scores(
        lexical_scores={},
        recalled_channels=(),
        relation_context=_ctx(),
    )
    assert scores == {}


def test_cjk_only_text_produces_lexical_scores() -> None:
    scores = match_touch_scores("谢谢")
    assert "warmth" in scores


def test_boundary_history_infers_boundary_touch() -> None:
    scores = graph_mediated_touch_scores(
        lexical_scores={},
        recalled_channels=(TraceChannel.BOUNDARY,),
        relation_context=_ctx(boundary_events=3),
    )
    assert "boundary" in scores
    assert scores["boundary"] > 0.0
