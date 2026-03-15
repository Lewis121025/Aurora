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


ORIENTATION_RISK_THRESHOLD = 3
ORIENTATION_STABILITY_THRESHOLD = 2


def plan_response(context: ExpressionContext) -> ResponseAct:
    relation = context.relation_context
    dominant = set(context.dominant_channels)
    risk_pressure = _orientation_risk_pressure(context.orientation_snapshot)
    stability_support = _orientation_stability_support(context.orientation_snapshot)

    if TraceChannel.BOUNDARY in dominant and relation.boundary_events >= relation.repair_events:
        return ResponseAct(
            move="boundary",
            tone="firm",
            response_channels=(TraceChannel.BOUNDARY, TraceChannel.COHERENCE),
        )
    if relation.boundary_events >= 3 and relation.repair_events == 0:
        return ResponseAct(
            move="boundary",
            tone="firm",
            response_channels=(TraceChannel.BOUNDARY, TraceChannel.COHERENCE),
        )
    if TraceChannel.REPAIR in dominant:
        return ResponseAct(
            move="repair",
            tone="gentle",
            response_channels=(TraceChannel.REPAIR, TraceChannel.WARMTH),
        )
    if (
        TraceChannel.DISTANCE in dominant
        and TraceChannel.HURT in dominant
        and relation.boundary_events > relation.resonance_events
        and relation.knot_count > 0
    ):
        return ResponseAct(
            move="silence",
            tone="quiet",
            response_channels=(TraceChannel.DISTANCE,),
        )
    if (
        context.has_knots
        and relation.repair_events * 2 < relation.boundary_events + relation.knot_count
    ):
        return ResponseAct(
            move="withhold",
            tone="guarded",
            response_channels=(TraceChannel.DISTANCE, TraceChannel.COHERENCE),
        )
    if TraceChannel.DISTANCE in dominant and relation.resonance_events <= relation.boundary_events:
        return ResponseAct(
            move="withhold",
            tone="guarded",
            response_channels=(TraceChannel.DISTANCE, TraceChannel.COHERENCE),
        )
    if risk_pressure and TraceChannel.HURT in dominant:
        return ResponseAct(
            move="withhold",
            tone="guarded",
            response_channels=(TraceChannel.DISTANCE, TraceChannel.COHERENCE),
        )
    if (
        TraceChannel.WARMTH in dominant or TraceChannel.RECOGNITION in dominant
    ) and relation.resonance_events + relation.thread_count >= relation.boundary_events:
        tone: Tone = "gentle"
        if stability_support:
            tone = "gentle"
        return ResponseAct(
            move="approach",
            tone=tone,
            response_channels=(TraceChannel.WARMTH, TraceChannel.RECOGNITION),
        )
    return ResponseAct(
        move="witness",
        tone="steady",
        response_channels=(TraceChannel.RECOGNITION, TraceChannel.COHERENCE),
    )


def _orientation_risk_pressure(snapshot: dict[str, object] | None) -> bool:
    if snapshot is None:
        return False
    world = snapshot.get("world")
    if not isinstance(world, dict):
        return False
    risk = world.get("risk")
    if not isinstance(risk, dict):
        return False
    count = risk.get("count", 0)
    return int(count) >= ORIENTATION_RISK_THRESHOLD if isinstance(count, (int, float)) else False


def _orientation_stability_support(snapshot: dict[str, object] | None) -> bool:
    if snapshot is None:
        return False
    world = snapshot.get("world")
    if not isinstance(world, dict):
        return False
    stability = world.get("stability")
    if not isinstance(stability, dict):
        return False
    count = stability.get("count", 0)
    return int(count) >= ORIENTATION_STABILITY_THRESHOLD if isinstance(count, (int, float)) else False
