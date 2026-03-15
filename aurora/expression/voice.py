from __future__ import annotations

from aurora.expression.context import ExpressionContext
from aurora.expression.response import ResponseAct
from aurora.expression.template_store import expression_text
from aurora.runtime.contracts import TraceChannel


def render_voice(context: ExpressionContext, act: ResponseAct) -> str:
    if act.move == "repair":
        variant = "history" if context.relation_context.repair_events > 0 else "default"
        return expression_text("repair", variant)
    if act.move == "approach":
        variant = "gratitude" if context.relation_context.thread_count > 0 else "default"
        if (
            variant == "default"
            and TraceChannel.RECOGNITION in context.dominant_channels
            and TraceChannel.WARMTH not in context.dominant_channels
        ):
            variant = "recognition"
        return expression_text("approach", variant)
    if act.move == "witness":
        if (
            TraceChannel.CURIOSITY in context.dominant_channels
            or TraceChannel.WONDER in context.dominant_channels
        ):
            return expression_text("witness", "curious")
        if (
            TraceChannel.HURT in context.dominant_channels
            or TraceChannel.BOUNDARY in context.dominant_channels
        ):
            return expression_text("witness", "hurt")
        return expression_text("witness")
    raise ValueError(f"unsupported voiced move: {act.move}")
