from __future__ import annotations

from aurora.being.touch import graph_mediated_touch_scores, match_touch_scores
from aurora.expression.context import ExpressionContext
from aurora.expression.render import render_response
from aurora.expression.response import plan_response
from aurora.relation.decision import RelationDecisionContext
from aurora.runtime.contracts import TraceChannel


def _relation_context(
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


def test_expression_planner_uses_boundary_pressure() -> None:
    act = plan_response(
        ExpressionContext(
            input_text="不要继续，边界在这里",
            relation_context=_relation_context(),
            dominant_channels=(TraceChannel.BOUNDARY, TraceChannel.COHERENCE),
            has_knots=False,
        )
    )

    assert act.move == "boundary"
    assert TraceChannel.BOUNDARY in act.response_channels


def test_expression_planner_can_choose_approach_from_warm_context() -> None:
    act = plan_response(
        ExpressionContext(
            input_text="谢谢你理解我",
            relation_context=_relation_context(resonance_events=1, thread_count=1),
            dominant_channels=(TraceChannel.WARMTH, TraceChannel.RECOGNITION),
            has_knots=False,
        )
    )

    assert act.move == "approach"
    assert TraceChannel.WARMTH in act.response_channels


def test_expression_renderer_produces_refusal_text_and_relation_edge() -> None:
    context = ExpressionContext(
        input_text="不要继续，边界在这里",
        relation_context=_relation_context(boundary_events=1),
        dominant_channels=(TraceChannel.BOUNDARY, TraceChannel.COHERENCE),
        has_knots=False,
    )
    act = plan_response(context)
    rendered = render_response(context, act)

    assert rendered.move == "boundary"
    assert "boundary" in rendered.text.lower() or "stopping" in rendered.text.lower()
    assert rendered.association_kind.value == "boundary"


def test_expression_renderer_produces_gentle_approach_text() -> None:
    context = ExpressionContext(
        input_text="谢谢你理解我",
        relation_context=_relation_context(resonance_events=2, thread_count=1),
        dominant_channels=(TraceChannel.WARMTH, TraceChannel.RECOGNITION),
        has_knots=False,
    )
    act = plan_response(context)
    rendered = render_response(context, act)

    assert rendered.move == "approach"
    assert "thread" in rendered.text.lower() or "close" in rendered.text.lower()
    assert rendered.response_traces


def test_touch_lexicon_drives_touch_scores() -> None:
    scores = match_touch_scores("谢谢你理解我，我想修复")

    assert scores["warmth"] > 0.0
    assert scores["insight"] > 0.0
    assert scores["repair"] > 0.0


def test_graph_mediated_touch_prefers_history_supported_touch() -> None:
    lexical = match_touch_scores("谢谢你理解我")
    cold = graph_mediated_touch_scores(
        lexical_scores=lexical,
        recalled_channels=(TraceChannel.BOUNDARY, TraceChannel.DISTANCE),
        relation_context=_relation_context(boundary_events=2, resonance_events=0, knot_count=1),
    )
    warm = graph_mediated_touch_scores(
        lexical_scores=lexical,
        recalled_channels=(TraceChannel.WARMTH, TraceChannel.RECOGNITION),
        relation_context=_relation_context(resonance_events=2, thread_count=1),
    )

    assert warm["warmth"] > cold["warmth"]


def test_expression_planner_uses_canonical_relation_context_for_withhold() -> None:
    act = plan_response(
        ExpressionContext(
            input_text="我还是有点受伤",
            relation_context=_relation_context(boundary_events=2, repair_events=0, knot_count=1),
            dominant_channels=(TraceChannel.HURT, TraceChannel.DISTANCE),
            has_knots=True,
        )
    )

    assert act.move in {"withhold", "silence"}


def test_expression_renderer_uses_contextual_witness_variant() -> None:
    context = ExpressionContext(
        input_text="我想知道之后会发生什么？",
        relation_context=_relation_context(resonance_events=1),
        dominant_channels=(TraceChannel.CURIOSITY, TraceChannel.COHERENCE),
        has_knots=False,
    )
    act = plan_response(context)
    rendered = render_response(context, act)

    assert rendered.move == "witness"
    assert "opening" in rendered.text.lower() or "carefully" in rendered.text.lower()
