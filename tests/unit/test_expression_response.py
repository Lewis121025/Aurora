from __future__ import annotations

from aurora.expression.context import ExpressionContext
from aurora.expression.render import render_response
from aurora.expression.response import plan_response
from aurora.runtime.contracts import TraceChannel


def test_expression_planner_uses_boundary_pressure() -> None:
    act = plan_response(
        ExpressionContext(
            input_text="不要继续，边界在这里",
            relation_snapshot={
                "boundary_events": 0,
                "repairability": 0.0,
                "trust": 0.0,
                "distance": 0.0,
            },
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
            relation_snapshot={
                "boundary_events": 0,
                "repairability": 0.0,
                "trust": 0.5,
                "distance": 0.0,
            },
            dominant_channels=(TraceChannel.WARMTH, TraceChannel.RECOGNITION),
            has_knots=False,
        )
    )

    assert act.move == "approach"
    assert TraceChannel.WARMTH in act.response_channels


def test_expression_renderer_produces_refusal_text_and_relation_edge() -> None:
    context = ExpressionContext(
        input_text="不要继续，边界在这里",
        relation_snapshot={
            "boundary_events": 1,
            "repairability": 0.0,
            "trust": 0.0,
            "distance": 0.0,
        },
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
        relation_snapshot={
            "boundary_events": 0,
            "repairability": 0.0,
            "trust": 0.4,
            "distance": 0.0,
        },
        dominant_channels=(TraceChannel.WARMTH, TraceChannel.RECOGNITION),
        has_knots=False,
    )
    act = plan_response(context)
    rendered = render_response(context, act)

    assert rendered.move == "approach"
    assert "thread" in rendered.text.lower() or "close" in rendered.text.lower()
    assert rendered.response_traces
