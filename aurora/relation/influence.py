from __future__ import annotations

from aurora.runtime.models import Tone


def tone_prefix(tone: Tone) -> str:
    if tone == "warm":
        return "I hear you."
    if tone == "cold":
        return "I heard that."
    if tone == "boundary":
        return "I need a boundary here."
    return "I am listening."
