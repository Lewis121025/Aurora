from __future__ import annotations

from uuid import uuid4

from aurora.memory.store import MemoryStore
from aurora.phases.transitions import phase_transition
from aurora.relation.store import RelationStore
from aurora.runtime.models import (
    ActivationView,
    AssocKind,
    AwakeOutcome,
    AuroraMove,
    BeingState,
    Phase,
    Speaker,
    Trace,
    TraceChannel,
    Turn,
)


def run_awake(
    relation_id: str,
    session_id: str,
    text: str,
    being: BeingState,
    memory_store: MemoryStore,
    relation_store: RelationStore,
    now_ts: float,
) -> AwakeOutcome:
    previous_phase = being.phase
    being.phase = Phase.AWAKE
    being.active_relation_id = relation_id

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
        surface=text[:160],
        vividness=min(1.0, 0.35 + len(text) / 320.0),
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

    activation = ActivationView(
        relation_state=relation_store.summarize_relation(relation_id),
        fragments=recalled,
        traces=tuple(
            trace
            for fragment in recalled
            for trace in memory_store.traces_for_fragment(fragment.fragment_id)
        ),
        chapters=memory_store.chapters_for_relation(relation_id),
        dominant_channels=memory_store.build_activation_channels(recalled + (user_fragment,)),
        boundary_required=TraceChannel.BOUNDARY in {trace.channel for trace in user_traces},
    )
    response_text, aurora_move = _policy(activation=activation, text=text)

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
        surface=response_text[:160],
        vividness=0.42,
        salience=0.44,
        unresolvedness=0.28 if aurora_move == "withhold" else 0.18,
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
        for channel, intensity in _traces_from_move(aurora_move)
    )
    memory_store.link_fragments(
        src_fragment_id=user_fragment.fragment_id,
        dst_fragment_id=aurora_fragment.fragment_id,
        kind=_edge_kind_from_move(aurora_move),
        weight=0.72,
        evidence=(user_turn.turn_id, aurora_turn.turn_id),
        now_ts=now_ts,
    )
    relation_store.record_exchange(
        relation_id=relation_id,
        user_turn_id=user_turn.turn_id,
        aurora_turn_id=aurora_turn.turn_id,
        user_channels=tuple(trace.channel for trace in user_traces),
        aurora_move=aurora_move,
        summary=f"{user_turn.text[:70]} -> {response_text[:70]}",
        now_ts=now_ts,
    )

    being.sleep_pressure = min(1.0, being.sleep_pressure + 0.08)
    being.continuity_pressure = min(1.0, being.continuity_pressure + 0.06)
    _drift_being_from_commit(being=being, user_traces=user_traces, aurora_move=aurora_move)

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
        activation=activation,
        transition=transition,
    )


def _policy(activation: ActivationView, text: str) -> tuple[str, AuroraMove]:
    lowered = text.lower()
    state = activation.relation_state
    boundary_tension_raw = state.get("boundary_tension", 0.0)
    distance_raw = state.get("distance", 0.0)
    boundary_tension = boundary_tension_raw if isinstance(boundary_tension_raw, float) else 0.0
    distance = distance_raw if isinstance(distance_raw, float) else 0.0
    if any(token in lowered for token in ("shut up", "get lost", "闭嘴", "滚")):
        return "I will stay quiet for now.", "silence"
    if activation.boundary_required or boundary_tension >= 0.45:
        return "I need to keep a clear boundary here.", "boundary"
    if "sorry" in lowered or "对不起" in text or "修复" in text:
        return "I can stay with a repair attempt if it remains steady.", "repair"
    if distance >= 0.35:
        return "I am still here, but I am keeping some distance.", "withhold"
    if activation.chapters:
        return (
            f"I can feel this touching an existing thread: {activation.chapters[-1].title}.",
            "approach",
        )
    return "I am here, and I am tracking what is taking shape between us.", "approach"


def _infer_traces(text: str) -> list[tuple[TraceChannel, float]]:
    lower = text.lower()
    traces: list[tuple[TraceChannel, float]] = [(TraceChannel.COHERENCE, 0.25)]
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


def _traces_from_move(aurora_move: AuroraMove) -> list[tuple[TraceChannel, float]]:
    if aurora_move == "boundary":
        return [(TraceChannel.BOUNDARY, 0.70), (TraceChannel.COHERENCE, 0.42)]
    if aurora_move == "repair":
        return [(TraceChannel.REPAIR, 0.66), (TraceChannel.WARMTH, 0.32)]
    if aurora_move == "withhold":
        return [(TraceChannel.DISTANCE, 0.48), (TraceChannel.COHERENCE, 0.32)]
    if aurora_move == "silence":
        return [(TraceChannel.DISTANCE, 0.38)]
    return [(TraceChannel.WARMTH, 0.28), (TraceChannel.RECOGNITION, 0.34)]


def _edge_kind_from_move(aurora_move: AuroraMove) -> AssocKind:
    if aurora_move == "boundary":
        return AssocKind.BOUNDARY
    if aurora_move == "repair":
        return AssocKind.REPAIR
    if aurora_move == "withhold":
        return AssocKind.CONTRAST
    return AssocKind.RELATION


def _estimate_unresolvedness(text: str) -> float:
    score = 0.35
    if any(token in text for token in ("为什么", "如何", "以后", "还会", "怎么办")):
        score += 0.20
    if any(token in text for token in ("?", "？")):
        score += 0.12
    if any(token in text for token in ("害怕", "失去", "矛盾", "边界", "不确定")):
        score += 0.18
    return min(1.0, score)


def _drift_being_from_commit(
    being: BeingState, user_traces: tuple[Trace, ...], aurora_move: AuroraMove
) -> None:
    channels = {trace.channel for trace in user_traces}
    self_updates = {"recognition": 0.0, "fragility": 0.0, "openness": 0.0, "agency": 0.0}
    world_updates = {"welcome": 0.0, "risk": 0.0, "mystery": 0.0, "stability": 0.0}
    if TraceChannel.RECOGNITION in channels:
        self_updates["recognition"] += 0.04
        world_updates["welcome"] += 0.03
    if TraceChannel.HURT in channels:
        self_updates["fragility"] += 0.05
        world_updates["risk"] += 0.05
    if TraceChannel.CURIOSITY in channels:
        world_updates["mystery"] += 0.04
    if TraceChannel.BOUNDARY in channels or aurora_move == "boundary":
        being.boundary_tension = min(1.0, being.boundary_tension + 0.08)
        world_updates["risk"] += 0.03
    if aurora_move in {"approach", "repair"}:
        self_updates["openness"] += 0.03
        self_updates["agency"] += 0.02
        world_updates["stability"] += 0.03
    if aurora_move == "withhold":
        self_updates["agency"] += 0.01
    being.drift(self_updates, world_updates)
