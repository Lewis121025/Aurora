from __future__ import annotations

from aurora.relation.boundaries import should_withhold


def choose_silence(text: str) -> bool:
    return should_withhold(text)
