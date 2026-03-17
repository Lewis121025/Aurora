"""Open loop helper。"""

from __future__ import annotations

from aurora.runtime.contracts import OpenLoop

_HALF_LIFE_HOURS = {
    "commitment": 168.0,
    "contradiction": 72.0,
    "unfinished_thread": 48.0,
    "unresolved_question": 24.0,
}


def current_urgency(loop: OpenLoop, now_ts: float) -> float:
    """按 loop 类型进行半衰期衰减。"""
    hours_elapsed = max(0.0, (now_ts - loop.updated_at) / 3600.0)
    half_life = _HALF_LIFE_HOURS[loop.loop_type]
    decay = float(0.5 ** (hours_elapsed / half_life))
    return float(loop.urgency * decay)


def top_open_loops(loops: tuple[OpenLoop, ...], now_ts: float, limit: int = 3) -> tuple[OpenLoop, ...]:
    """选出当前最重要的 active loops。"""
    active = [loop for loop in loops if loop.status == "active"]
    ordered = sorted(active, key=lambda item: current_urgency(item, now_ts), reverse=True)
    return tuple(ordered[:limit])


def to_prompt_segment(loops: tuple[OpenLoop, ...], now_ts: float) -> str:
    """将 open loops 投影为 prompt 片段。"""
    ranked = top_open_loops(loops, now_ts)
    if not ranked:
        return "[OPEN_LOOPS]\nnone"

    lines = ["[OPEN_LOOPS]"]
    for loop in ranked:
        lines.append(
            f"- {loop.loop_type}: {loop.summary} (urgency={current_urgency(loop, now_ts):.2f})"
        )
    return "\n".join(lines)
