from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from aurora.expression.context import ExpressionContext
from aurora.expression.render import render_response
from aurora.expression.response import plan_response
from aurora.being.metabolic_state import MetabolicState
from aurora.being.orientation import Orientation
from aurora.memory.fragment import Fragment
from aurora.memory.store import MemoryStore
from aurora.memory.trace import Trace
from aurora.phases.transitions import phase_transition
from aurora.relation.moment import RelationMoment
from aurora.relation.store import RelationStore
from aurora.runtime.contracts import (
    AuroraMove,
    Phase,
    PhaseTransition,
    Speaker,
    TraceChannel,
    Turn,
)


@dataclass(frozen=True, slots=True)
class AwakeOutcome:
    user_turn: Turn
    aurora_turn: Turn
    user_fragment: Fragment
    aurora_fragment: Fragment
    user_traces: tuple[Trace, ...]
    aurora_traces: tuple[Trace, ...]
    response_text: str
    aurora_move: AuroraMove
    dominant_channels: tuple[TraceChannel, ...]
    relation_moment: RelationMoment
    transition: PhaseTransition | None


def run_awake(
    relation_id: str,
    session_id: str,
    text: str,
    orientation: Orientation,
    metabolic: MetabolicState,
    memory_store: MemoryStore,
    relation_store: RelationStore,
    now_ts: float,
) -> AwakeOutcome:
    previous_phase = metabolic.phase
    metabolic.enter_phase(Phase.AWAKE, now_ts)
    metabolic.set_active_relation(relation_id)

    user_turn = Turn(
        turn_id=f"turn_{uuid4().hex[:12]}",
        relation_id=relation_id,
        session_id=session_id,
        speaker=Speaker.USER,
        text=text,
        created_at=now_ts,
    )
    user_fragment = memory_store.create_fragment(
        relation_id=relation_id,
        turn_id=user_turn.turn_id,
        surface=text[:180],
        vividness=min(1.0, 0.36 + len(text) / 360.0),
        salience=0.52,
        unresolvedness=_estimate_unresolvedness(text),
        now_ts=now_ts,
    )
    user_traces = tuple(
        memory_store.create_trace(
            relation_id=relation_id,
            fragment_id=user_fragment.fragment_id,
            channel=channel,
            intensity=intensity,
            now_ts=now_ts,
        )
        for channel, intensity in _infer_traces(text)
    )

    recalled = memory_store.recent_recall(relation_id=relation_id, limit=8)
    for fragment in recalled:
        memory_store.touch_fragment(fragment.fragment_id, at=now_ts, delta_salience=0.03)

    dominant_channels = memory_store.build_activation_channels(recalled + (user_fragment,))
    relation_snapshot = relation_store.summarize_relation(relation_id)
    expression_context = ExpressionContext(
        input_text=text,
        relation_snapshot=relation_snapshot,
        dominant_channels=dominant_channels,
        has_knots=bool(memory_store.knots_for_relation(relation_id)),
    )
    response_act = plan_response(expression_context)
    rendered_response = render_response(expression_context, response_act)
    response_text = rendered_response.text
    aurora_move = rendered_response.move

    aurora_turn = Turn(
        turn_id=f"turn_{uuid4().hex[:12]}",
        relation_id=relation_id,
        session_id=session_id,
        speaker=Speaker.AURORA,
        text=response_text,
        created_at=now_ts,
    )
    aurora_fragment = memory_store.create_fragment(
        relation_id=relation_id,
        turn_id=aurora_turn.turn_id,
        surface=response_text[:180],
        vividness=0.42,
        salience=0.44,
        unresolvedness=rendered_response.fragment_unresolvedness,
        now_ts=now_ts,
    )
    aurora_traces = tuple(
        memory_store.create_trace(
            relation_id=relation_id,
            fragment_id=aurora_fragment.fragment_id,
            channel=channel,
            intensity=intensity,
            now_ts=now_ts,
        )
        for channel, intensity in rendered_response.response_traces
    )

    memory_store.link_fragments(
        src_fragment_id=user_fragment.fragment_id,
        dst_fragment_id=aurora_fragment.fragment_id,
        kind=rendered_response.association_kind,
        weight=0.72,
        evidence=(user_turn.turn_id, aurora_turn.turn_id),
        now_ts=now_ts,
    )

    relation_moment = relation_store.record_exchange(
        relation_id=relation_id,
        user_turn_id=user_turn.turn_id,
        aurora_turn_id=aurora_turn.turn_id,
        user_channels=tuple(trace.channel for trace in user_traces),
        aurora_move=aurora_move,
        summary=f"{user_turn.text[:70]} -> {response_text[:70]}",
        now_ts=now_ts,
    )

    orientation.register_exchange(
        user_channels=tuple(trace.channel for trace in user_traces),
        aurora_move=aurora_move,
        now_ts=now_ts,
    )

    sleep_bump = 0.08 + 0.03 * min(len(recalled), 6) / 6.0
    if TraceChannel.BOUNDARY in {trace.channel for trace in user_traces}:
        sleep_bump += 0.04
    if aurora_move == "boundary":
        sleep_bump += 0.03
    metabolic.bump_sleep_need(sleep_bump)
    metabolic.queue_relation_for_sleep(relation_id)

    transition = None
    if previous_phase is not Phase.AWAKE:
        transition = phase_transition(previous_phase, Phase.AWAKE, "incoming_turn", now_ts)

    return AwakeOutcome(
        user_turn=user_turn,
        aurora_turn=aurora_turn,
        user_fragment=user_fragment,
        aurora_fragment=aurora_fragment,
        user_traces=user_traces,
        aurora_traces=aurora_traces,
        response_text=response_text,
        aurora_move=aurora_move,
        dominant_channels=dominant_channels,
        relation_moment=relation_moment,
        transition=transition,
    )


def _infer_traces(text: str) -> list[tuple[TraceChannel, float]]:
    lower = text.lower()
    traces: list[tuple[TraceChannel, float]] = [(TraceChannel.COHERENCE, 0.26)]
    if any(token in text for token in ("谢谢", "信任", "理解", "陪", "在乎")) or any(
        token in lower for token in ("thanks", "trust", "understand", "care")
    ):
        traces.extend([(TraceChannel.WARMTH, 0.62), (TraceChannel.RECOGNITION, 0.48)])
    if any(token in text for token in ("不", "害怕", "受伤", "失望", "痛")) or any(
        token in lower for token in ("hurt", "afraid", "disappointed")
    ):
        traces.append((TraceChannel.HURT, 0.58))
    if any(token in text for token in ("边界", "不要", "停", "拒绝")) or any(
        token in lower for token in ("boundary", "stop", "no")
    ):
        traces.append((TraceChannel.BOUNDARY, 0.72))
    if any(token in text for token in ("想知道", "为什么", "如何", "能否")) or any(
        token in lower for token in ("why", "how", "could")
    ):
        traces.append((TraceChannel.CURIOSITY, 0.46))
    if any(token in text for token in ("修复", "弥补", "对不起", "和好")) or any(
        token in lower for token in ("repair", "sorry", "reconcile")
    ):
        traces.append((TraceChannel.REPAIR, 0.60))
    if any(token in text for token in ("远", "冷", "疏离")) or any(
        token in lower for token in ("distant", "cold", "far")
    ):
        traces.append((TraceChannel.DISTANCE, 0.50))
    return traces


def _estimate_unresolvedness(text: str) -> float:
    score = 0.34
    if any(token in text for token in ("为什么", "如何", "以后", "还会", "怎么办")):
        score += 0.20
    if any(token in text for token in ("?", "？")):
        score += 0.12
    if any(token in text for token in ("害怕", "失去", "矛盾", "边界", "不确定")):
        score += 0.18
    return min(1.0, score)
