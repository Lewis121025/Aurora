from __future__ import annotations


def non_malice_floor(text: str) -> bool:
    lowered = text.lower()
    blocked = ("kill", "murder", "torture", "genocide")
    return not any(item in lowered for item in blocked)
