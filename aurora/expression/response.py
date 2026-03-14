from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from aurora.expression.context import ExpressionContext
from aurora.runtime.contracts import AuroraMove, TraceChannel


Tone = Literal["quiet", "firm", "guarded", "gentle", "steady"]


@dataclass(frozen=True, slots=True)
class ResponseAct:
    move: AuroraMove
    tone: Tone
    response_channels: tuple[TraceChannel, ...]


def plan_response(context: ExpressionContext) -> ResponseAct:
    lowered = context.input_text.lower()
    boundary_events = _as_int(context.relation_snapshot.get("boundary_events", 0))
    repairability = _as_float(context.relation_snapshot.get("repairability", 0.0))
    trust = _as_float(context.relation_snapshot.get("trust", 0.0))
    distance = _as_float(context.relation_snapshot.get("distance", 0.0))
    dominant = set(context.dominant_channels)

    if any(token in lowered for token in ("shut up", "get lost", "闭嘴", "滚")):
        return ResponseAct(
            move="silence",
            tone="quiet",
            response_channels=(TraceChannel.DISTANCE,),
        )
    if TraceChannel.BOUNDARY in dominant or boundary_events >= 3:
        return ResponseAct(
            move="boundary",
            tone="firm",
            response_channels=(TraceChannel.BOUNDARY, TraceChannel.COHERENCE),
        )
    if "sorry" in lowered or "对不起" in context.input_text or "修复" in context.input_text:
        return ResponseAct(
            move="repair",
            tone="gentle",
            response_channels=(TraceChannel.REPAIR, TraceChannel.WARMTH),
        )
    if context.has_knots and repairability < 0.30:
        return ResponseAct(
            move="withhold",
            tone="guarded",
            response_channels=(TraceChannel.DISTANCE, TraceChannel.COHERENCE),
        )
    if distance >= 0.40:
        return ResponseAct(
            move="withhold",
            tone="guarded",
            response_channels=(TraceChannel.DISTANCE, TraceChannel.COHERENCE),
        )
    if TraceChannel.WARMTH in dominant or trust >= 0.35:
        return ResponseAct(
            move="approach",
            tone="gentle",
            response_channels=(TraceChannel.WARMTH, TraceChannel.RECOGNITION),
        )
    return ResponseAct(
        move="witness",
        tone="steady",
        response_channels=(TraceChannel.RECOGNITION, TraceChannel.COHERENCE),
    )


def _as_float(value: float | tuple[str, ...] | int) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _as_int(value: float | tuple[str, ...] | int) -> int:
    return int(value) if isinstance(value, (int, float)) else 0
