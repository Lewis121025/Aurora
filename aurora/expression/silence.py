from __future__ import annotations

from aurora.expression.context import ExpressionContext
from aurora.expression.template_store import expression_text
from aurora.expression.response import Tone
from aurora.runtime.contracts import AuroraMove, TraceChannel


def render_refusal(context: ExpressionContext, move: AuroraMove, tone: Tone) -> str:
    if move == "silence":
        variant = "firm" if context.relation_context.boundary_events > 1 else tone
        return expression_text("silence", variant)
    if move == "boundary":
        if context.has_knots or context.relation_context.knot_count > 0:
            return expression_text("boundary", "distance")
        if TraceChannel.BOUNDARY in context.dominant_channels:
            return expression_text("boundary", "stop")
        if context.relation_context.boundary_events > context.relation_context.repair_events:
            return expression_text("boundary", "hold")
        return expression_text("boundary")
    if move == "withhold":
        if context.has_knots:
            return expression_text("withhold", "knot")
        if context.relation_context.boundary_events > context.relation_context.resonance_events:
            return expression_text("withhold", "distance")
        return expression_text("withhold", "default")
    raise ValueError(f"unsupported refusal move: {move}")
