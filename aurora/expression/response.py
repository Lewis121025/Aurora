from __future__ import annotations

from aurora.expression.silence import choose_silence
from aurora.expression.voice import base_voice_prefix
from aurora.relation.influence import tone_prefix
from aurora.runtime.models import ExpressionContext


def _stance_from_touch(touch_modes: tuple[str, ...]) -> str:
    if "boundary" in touch_modes:
        return "I need to keep this clear and respectful."
    if "hurt" in touch_modes:
        return "That lands heavily with me."
    if "warmth" in touch_modes:
        return "I feel the warmth in this."
    if "insight" in touch_modes:
        return "I can feel this changing how I understand things."
    if "curiosity" in touch_modes:
        return "I want to stay with this and explore it."
    return "I am staying with this moment."


def render_response(context: ExpressionContext) -> str:
    if choose_silence(context.input_text):
        return "I will stay quiet for now."

    if context.relation_tone == "boundary" and context.relation_strength >= 0.6:
        return "I need to pause here. I will not continue this line of interaction."

    if context.relation_tone == "cold" and context.relation_strength >= 0.7:
        return "I hear you, but I need more care in how we speak to continue well."

    prefix = tone_prefix(context.relation_tone)
    base = base_voice_prefix()
    stance = _stance_from_touch(context.touch_modes)
    if context.memory_snippets:
        memory_hint = context.memory_snippets[-1]
        return f"{base} {prefix} {stance} I remember this thread: {memory_hint}"
    return f"{base} {prefix} {stance}"
