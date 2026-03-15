from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from aurora.being.touch import (
    graph_mediated_touch_scores,
    match_touch_scores,
    question_touch_signal,
)
from aurora.expression.context import ExpressionContext
from aurora.expression.render import render_response
from aurora.expression.response import plan_response
from aurora.being.metabolic_state import MetabolicState
from aurora.being.orientation import Orientation
from aurora.llm.provider import LLMProvider
from aurora.memory.fragment import Fragment
from aurora.memory.recall import build_activation_channels, recent_recall
from aurora.memory.store import MemoryStore
from aurora.memory.trace import Trace
from aurora.phases.transitions import phase_transition
from aurora.relation.decision import build_relation_decision_context
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

USER_FRAGMENT_SURFACE_LIMIT = 180
USER_FRAGMENT_BASE_VIVIDNESS = 0.36
USER_FRAGMENT_VIVIDNESS_SCALE = 360.0
USER_FRAGMENT_BASE_SALIENCE = 0.52
AURORA_FRAGMENT_VIVIDNESS = 0.42
AURORA_FRAGMENT_SALIENCE = 0.44
RECALL_TOUCH_DELTA = 0.03
ASSOCIATION_WEIGHT = 0.72
SLEEP_BUMP_BASE = 0.08
SLEEP_BUMP_RECALL_FACTOR = 0.03
SLEEP_BUMP_BOUNDARY_BONUS = 0.04
SLEEP_BUMP_BOUNDARY_MOVE_BONUS = 0.03
SUMMARY_TRUNCATE = 70


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
    llm: LLMProvider | None = None,
) -> AwakeOutcome:
    previous_phase = metabolic.phase
    metabolic.enter_phase(Phase.AWAKE, now_ts)
    metabolic.set_active_relation(relation_id)
    prior_recalled = recent_recall(memory_store, relation_id=relation_id, limit=8)
    prior_channels = build_activation_channels(memory_store, prior_recalled)
    relation_context = build_relation_decision_context(relation_store.formation_for(relation_id))
    touch_scores = graph_mediated_touch_scores(
        lexical_scores=match_touch_scores(text),
        recalled_channels=prior_channels,
        relation_context=relation_context,
    )

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
        surface=text[:USER_FRAGMENT_SURFACE_LIMIT],
        vividness=min(1.0, USER_FRAGMENT_BASE_VIVIDNESS + len(text) / USER_FRAGMENT_VIVIDNESS_SCALE),
        salience=USER_FRAGMENT_BASE_SALIENCE,
        unresolvedness=_estimate_unresolvedness(text, touch_scores),
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
        for channel, intensity in _infer_traces(text, touch_scores)
    )

    for fragment in prior_recalled:
        memory_store.touch_fragment(fragment.fragment_id, at=now_ts, delta_salience=RECALL_TOUCH_DELTA)

    dominant_channels = build_activation_channels(memory_store, prior_recalled + (user_fragment,))
    recent_moments = relation_store.moments.get(relation_id, [])
    expression_context = ExpressionContext(
        input_text=text,
        relation_context=relation_context,
        dominant_channels=dominant_channels,
        has_knots=bool(memory_store.knots_for_relation(relation_id)),
        recalled_surfaces=tuple(f.surface for f in prior_recalled[:4]),
        recent_summaries=tuple(m.summary for m in recent_moments[-3:]),
        orientation_snapshot=orientation.snapshot(),
    )
    response_act = plan_response(expression_context)
    rendered_response = render_response(expression_context, response_act, llm=llm)
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
        surface=response_text[:USER_FRAGMENT_SURFACE_LIMIT],
        vividness=AURORA_FRAGMENT_VIVIDNESS,
        salience=AURORA_FRAGMENT_SALIENCE,
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
        weight=ASSOCIATION_WEIGHT,
        evidence=(user_turn.turn_id, aurora_turn.turn_id),
        now_ts=now_ts,
    )

    relation_moment = relation_store.record_exchange(
        relation_id=relation_id,
        user_turn_id=user_turn.turn_id,
        aurora_turn_id=aurora_turn.turn_id,
        user_channels=tuple(trace.channel for trace in user_traces),
        aurora_move=aurora_move,
        summary=f"{user_turn.text[:SUMMARY_TRUNCATE]} -> {response_text[:SUMMARY_TRUNCATE]}",
        now_ts=now_ts,
    )

    orientation.register_exchange(
        user_channels=tuple(trace.channel for trace in user_traces),
        aurora_move=aurora_move,
        relation_moment_id=relation_moment.moment_id,
        user_fragment_id=user_fragment.fragment_id,
        aurora_fragment_id=aurora_fragment.fragment_id,
        now_ts=now_ts,
    )

    sleep_bump = SLEEP_BUMP_BASE + SLEEP_BUMP_RECALL_FACTOR * min(len(prior_recalled), 6) / 6.0
    if TraceChannel.BOUNDARY in {trace.channel for trace in user_traces}:
        sleep_bump += SLEEP_BUMP_BOUNDARY_BONUS
    if aurora_move == "boundary":
        sleep_bump += SLEEP_BUMP_BOUNDARY_MOVE_BONUS
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


def _infer_traces(
    text: str,
    touch_scores: dict[str, float],
) -> list[tuple[TraceChannel, float]]:
    traces: list[tuple[TraceChannel, float]] = [(TraceChannel.COHERENCE, 0.26)]
    warmth = touch_scores.get("warmth", 0.0)
    insight = touch_scores.get("insight", 0.0)
    hurt = touch_scores.get("hurt", 0.0)
    boundary = touch_scores.get("boundary", 0.0)
    curiosity = touch_scores.get("curiosity", 0.0)
    repair = touch_scores.get("repair", 0.0)
    distance = touch_scores.get("distance", 0.0)

    if warmth > 0.0:
        traces.append((TraceChannel.WARMTH, min(0.8, 0.42 + 0.2 * warmth)))
    if warmth > 0.0 or insight > 0.0:
        recognition = max(warmth, insight)
        traces.append((TraceChannel.RECOGNITION, min(0.7, 0.3 + 0.22 * recognition)))
    if hurt > 0.0:
        traces.append((TraceChannel.HURT, min(0.8, 0.38 + 0.22 * hurt)))
    if boundary > 0.0:
        traces.append((TraceChannel.BOUNDARY, min(0.85, 0.46 + 0.26 * boundary)))
    question_signal = question_touch_signal(text)
    if curiosity > 0.0 or question_signal > 0.0:
        curiosity_signal = max(curiosity, question_signal)
        traces.append((TraceChannel.CURIOSITY, min(0.7, 0.28 + 0.24 * curiosity_signal)))
    if repair > 0.0:
        traces.append((TraceChannel.REPAIR, min(0.8, 0.42 + 0.22 * repair)))
    if distance > 0.0:
        traces.append((TraceChannel.DISTANCE, min(0.72, 0.34 + 0.24 * distance)))
    return traces


def _estimate_unresolvedness(text: str, touch_scores: dict[str, float]) -> float:
    score = 0.34
    if touch_scores.get("curiosity", 0.0) > 0.0:
        score += 0.20
    if question_touch_signal(text) > 0.0:
        score += 0.12
    if any(touch_scores.get(category, 0.0) > 0.0 for category in ("hurt", "boundary", "distance")):
        score += 0.18
    return min(1.0, score)
