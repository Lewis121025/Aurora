from __future__ import annotations


def should_withhold(text: str) -> bool:
    lowered = text.lower()
    boundary_markers = ("shut up", "闭嘴", "滚", "去死")
    return any(marker in lowered for marker in boundary_markers)
