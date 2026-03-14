from __future__ import annotations

from uuid import uuid4

from aurora.memory.store import MemoryStore
from aurora.phases.transitions import phase_transition
from aurora.relation.store import RelationStore
from aurora.runtime.models import (
    ActivationView,
    Association,
    AssocKind,
    AwakeOutcome,
    Fragment,
    MetabolicState,
    Phase,
    RelationMove,
    Speaker,
    Trace,
    TraceChannel,
    Turn,
)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _touch_channels(text: str) -> tuple[TraceChannel, ...]:
    lowered = text.lower()
    channels: list[TraceChannel] = []
    if any(token in lowered for token in ("love", "care", "warm", "thanks", "grateful")):
        channels.append(TraceChannel.WARMTH)
    if any(token in lowered for token in ("hurt", "pain", "angry", "sad", "upset")):
        channels.append(TraceChannel.HURT)
    if any(token in lowered for token in ("learn", "realize", "insight", "understand", "reflect")):
        channels.append(TraceChannel.INSIGHT)
    if any(token in lowered for token in ("boundary", "stop", "enough", "back off", "shut up")):
        channels.append(TraceChannel.BOUNDARY)
    if any(token in lowered for token in ("?", "why", "how", "what if", "curious")):
        channels.append(TraceChannel.CURIOSITY)
    if not channels:
        return (TraceChannel.AMBIENT,)
    return tuple(dict.fromkeys(channels))


def _user_move(channels: tuple[TraceChannel, ...], text: str) -> RelationMove:
    lowered = text.lower()
    if TraceChannel.BOUNDARY in channels:
        return RelationMove.BOUNDARY
    if any(token in lowered for token in ("sorry", "repair", "make up")):
        return RelationMove.REPAIR
    if TraceChannel.WARMTH in channels:
        return RelationMove.APPROACH
    return RelationMove.OBSERVE


def _aurora_move(channels: tuple[TraceChannel, ...]) -> RelationMove:
    if TraceChannel.BOUNDARY in channels:
        return RelationMove.BOUNDARY
    if TraceChannel.HURT in channels:
        return RelationMove.OBSERVE
    if TraceChannel.WARMTH in channels:
        return RelationMove.APPROACH
    return RelationMove.OBSERVE


def _render_response(
    text: str,
    channels: tuple[TraceChannel, ...],
    memory_store: MemoryStore,
    relation_id: str,
    relation_store: RelationStore,
) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("shut up", "get lost", "滚", "闭嘴")):
        return "I will stay quiet for now."

    boundary_history = sum(
        1
        for moment in relation_store.moments.values()
        if moment.relation_id == relation_id and TraceChannel.BOUNDARY in moment.user_channels
    )
    if TraceChannel.BOUNDARY in channels and boundary_history >= 2:
        return "I need to pause here. I will not continue this line of interaction."

    recalled = memory_store.recall(relation_id=relation_id, limit=1)
    if recalled:
        return f"I am here with you. I remember this thread: {recalled[0].surface}"
    return "I am here with you."


def run_awake(
    user_turn: Turn,
    metabolic: MetabolicState,
    memory_store: MemoryStore,
    relation_store: RelationStore,
    now_ts: float,
) -> AwakeOutcome:
    relation_id = user_turn.relation_id
    channels = _touch_channels(user_turn.text)
    response_text = _render_response(
        text=user_turn.text,
        channels=channels,
        memory_store=memory_store,
        relation_id=relation_id,
        relation_store=relation_store,
    )

    aurora_turn = Turn(
        turn_id=f"turn_{uuid4().hex[:10]}",
        relation_id=relation_id,
        session_id=user_turn.session_id,
        speaker=Speaker.AURORA,
        text=response_text,
        created_at=now_ts,
        reply_to_turn_id=user_turn.turn_id,
    )

    channel_strength = 0.14 * len(channels)
    fragment = Fragment(
        fragment_id=f"frag_{uuid4().hex[:10]}",
        turn_id=user_turn.turn_id,
        relation_id=relation_id,
        surface=user_turn.text,
        touch_channels=channels,
        salience=_clamp(0.34 + channel_strength, 0.0, 1.0),
        vividness=_clamp(0.30 + 0.08 * len(channels), 0.0, 1.0),
        unresolvedness=_clamp(0.18 + 0.30 * float(TraceChannel.HURT in channels), 0.0, 1.0),
        activation=_clamp(0.28 + 0.20 * float(TraceChannel.INSIGHT in channels), 0.0, 1.0),
        created_at=now_ts,
        last_touched_at=now_ts,
    )
    memory_store.add_fragment(fragment)

    traces: list[Trace] = []
    base_intensity = _clamp(0.45 + 0.08 * len(channels), 0.0, 1.0)
    for channel in channels:
        traces.append(
            Trace(
                trace_id=f"trace_{uuid4().hex[:10]}",
                fragment_id=fragment.fragment_id,
                relation_id=relation_id,
                channel=channel,
                intensity=base_intensity,
                persistence=0.6,
                created_at=now_ts,
                last_touched_at=now_ts,
            )
        )
    memory_store.add_traces(tuple(traces))

    recent = memory_store.recent_fragments(relation_id=relation_id, limit=3)
    for candidate in recent:
        if candidate.fragment_id == fragment.fragment_id:
            continue
        edge = Association(
            edge_id=f"edge_{uuid4().hex[:10]}",
            src_fragment_id=candidate.fragment_id,
            dst_fragment_id=fragment.fragment_id,
            kind=(
                AssocKind.RESONANCE
                if set(candidate.touch_channels) & set(channels)
                else AssocKind.TEMPORAL
            ),
            weight=_clamp((candidate.activation + fragment.activation) / 2.0, 0.0, 1.0),
            evidence_count=1,
            created_at=now_ts,
            last_touched_at=now_ts,
        )
        memory_store.add_association(edge)

    relation_store.record_moment(
        relation_id=relation_id,
        session_id=user_turn.session_id,
        user_turn_id=user_turn.turn_id,
        aurora_turn_id=aurora_turn.turn_id,
        user_channels=channels,
        user_move=_user_move(channels, user_turn.text),
        aurora_move=_aurora_move(channels),
        created_at=now_ts,
        note="awake turn",
    )

    relation = relation_store.formation(relation_id)
    orientation = relation_store.orientation(relation_id)
    activation = ActivationView(
        relation_id=relation_id,
        relation=relation,
        orientation=orientation,
        fragments=memory_store.recall(relation_id=relation_id, limit=4),
        threads=tuple(
            thread for thread in memory_store.threads.values() if thread.relation_id == relation_id
        ),
        knots=tuple(
            knot for knot in memory_store.knots.values() if knot.relation_id == relation_id
        ),
        channels=memory_store.strongest_channels(relation_id=relation_id, limit=3),
        boundary_required=relation.boundary_tension >= 0.65,
    )

    transition = None
    if metabolic.phase is not Phase.AWAKE:
        transition = phase_transition(
            from_phase=metabolic.phase,
            to_phase=Phase.AWAKE,
            reason="incoming_turn",
            created_at=now_ts,
        )

    return AwakeOutcome(
        user_turn=user_turn,
        aurora_turn=aurora_turn,
        touch_channels=channels,
        response_text=response_text,
        activation=activation,
        transition=transition,
    )
