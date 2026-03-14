from __future__ import annotations

from dataclasses import dataclass

from aurora.expression.context import ExpressionContext
from aurora.expression.response import ResponseAct
from aurora.expression.silence import render_refusal
from aurora.expression.voice import render_voice
from aurora.runtime.contracts import AssocKind, AuroraMove, TraceChannel


@dataclass(frozen=True, slots=True)
class RenderedResponse:
    move: AuroraMove
    text: str
    response_traces: tuple[tuple[TraceChannel, float], ...]
    association_kind: AssocKind
    fragment_unresolvedness: float


def render_response(context: ExpressionContext, act: ResponseAct) -> RenderedResponse:
    if act.move in {"silence", "boundary", "withhold"}:
        text = render_refusal(context=context, move=act.move, tone=act.tone)
    else:
        text = render_voice(context=context, act=act)
    return RenderedResponse(
        move=act.move,
        text=text,
        response_traces=_trace_intensities(act.response_channels),
        association_kind=_edge_kind_from_move(act.move),
        fragment_unresolvedness=0.24 if act.move == "withhold" else 0.16,
    )


def _trace_intensities(
    channels: tuple[TraceChannel, ...],
) -> tuple[tuple[TraceChannel, float], ...]:
    intensity_map = {
        TraceChannel.BOUNDARY: 0.70,
        TraceChannel.COHERENCE: 0.42,
        TraceChannel.REPAIR: 0.66,
        TraceChannel.WARMTH: 0.32,
        TraceChannel.DISTANCE: 0.48,
        TraceChannel.RECOGNITION: 0.34,
    }
    fallback = {
        TraceChannel.CURIOSITY: 0.30,
        TraceChannel.HURT: 0.34,
        TraceChannel.WONDER: 0.28,
    }
    return tuple(
        (channel, intensity_map.get(channel, fallback.get(channel, 0.28))) for channel in channels
    )


def _edge_kind_from_move(aurora_move: AuroraMove) -> AssocKind:
    if aurora_move == "boundary":
        return AssocKind.BOUNDARY
    if aurora_move == "repair":
        return AssocKind.REPAIR
    if aurora_move == "withhold":
        return AssocKind.CONTRAST
    return AssocKind.RELATION
