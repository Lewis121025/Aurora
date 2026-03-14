from __future__ import annotations

from aurora.expression.context import ExpressionContext
from aurora.expression.response import Tone
from aurora.runtime.contracts import AuroraMove


def render_refusal(context: ExpressionContext, move: AuroraMove, tone: Tone) -> str:
    if move == "silence":
        return "I will remain quiet here." if tone == "quiet" else "I am not speaking further here."
    if move == "boundary":
        if "不要" in context.input_text or "stop" in context.input_text.lower():
            return "I am holding the boundary and stopping here."
        return "I need to hold a boundary and not move past it."
    if move == "withhold":
        if context.has_knots:
            return "I am still here, but I need to keep distance for now."
        return "I am not leaving, but I am keeping some distance."
    raise ValueError(f"unsupported refusal move: {move}")
