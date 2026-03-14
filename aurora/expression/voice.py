from __future__ import annotations

from aurora.expression.context import ExpressionContext
from aurora.expression.response import ResponseAct


def render_voice(context: ExpressionContext, act: ResponseAct) -> str:
    if act.move == "repair":
        if "sorry" in context.input_text.lower() or "对不起" in context.input_text:
            return "I can stay with the repair attempt, but slowly."
        return "I can try to repair this, but I need to do it carefully."
    if act.move == "approach":
        if "谢谢" in context.input_text or "trust" in context.input_text.lower():
            return "I can feel the thread here and I want to stay with it."
        return "I want to stay close to what is opening here."
    if act.move == "witness":
        return "I am staying with what is present, without flattening it."
    raise ValueError(f"unsupported voiced move: {act.move}")
