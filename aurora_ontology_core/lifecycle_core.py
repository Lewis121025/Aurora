from __future__ import annotations

from dataclasses import replace
from typing import Callable
from uuid import uuid4

from core_models import (
    ActivationBundle,
    AwakeDraft,
    AuroraMove,
    BeingState,
    InteractionTurn,
    Phase,
    Speaker,
    TurnCommit,
    TraceChannel,
)
from memory_core import MemoryGraph, extract_tags
from relation_core import RelationLedger


ResponsePolicy = Callable[[ActivationBundle, str], tuple[str, AuroraMove]]


class AuroraCore:
    def __init__(self) -> None:
        self.memory = MemoryGraph()
        self.relations = RelationLedger()
        self.being = BeingState()

    def awake(self, relation_id: str, session_id: str, text: str, now_ts: float, policy: ResponsePolicy) -> TurnCommit:
        self.being.phase = Phase.AWAKE
        draft = self.prepare_awake(relation_id=relation_id, session_id=session_id, text=text, now_ts=now_ts)
        response_text, aurora_move = policy(draft.activation, draft.response_hint)
        return self.commit_awake(draft=draft, session_id=session_id, response_text=response_text, aurora_move=aurora_move, now_ts=now_ts)

    def prepare_awake(self, relation_id: str, session_id: str, text: str, now_ts: float) -> AwakeDraft:
        user_turn = InteractionTurn(
            turn_id=f"turn_{uuid4().hex[:12]}",
            relation_id=relation_id,
            session_id=session_id,
            speaker=Speaker.USER,
            text=text,
            created_at=now_ts,
        )
        tags = extract_tags(text)
        user_fragment = self.memory.create_fragment(
            relation_id=relation_id,
            turn_id=user_turn.turn_id,
            surface=text[:160],
            tags=tags,
            vividness=min(1.0, 0.35 + len(text) / 320.0),
            salience=0.52,
            unresolvedness=self._estimate_unresolvedness(text),
            now_ts=now_ts,
        )
        user_traces = tuple(
            self.memory.create_trace(
                relation_id=relation_id,
                fragment_id=user_fragment.fragment_id,
                channel=channel,
                intensity=intensity,
                now_ts=now_ts,
            )
            for channel, intensity in self._infer_traces(text)
        )

        recalled = self.memory.recent_recall(relation_id=relation_id, limit=8)
        for fragment in recalled:
            self.memory.touch_fragment(fragment.fragment_id, at=now_ts, delta_salience=0.03)

        activation = ActivationBundle(
            relation_state=self.relations.summarize_relation(relation_id),
            fragments=recalled,
            traces=tuple(trace for fragment in recalled for trace in self.memory.traces_for_fragment(fragment.fragment_id)),
            chapters=self.memory.chapters_for_relation(relation_id),
            dominant_channels=self.memory.build_activation_channels(recalled + (user_fragment,)),
            boundary_required=TraceChannel.BOUNDARY in {trace.channel for trace in user_traces},
        )
        response_hint = self._hint_from_activation(text=text, activation=activation)
        self.being.sleep_pressure = min(1.0, self.being.sleep_pressure + 0.08)
        self.being.continuity_pressure = min(1.0, self.being.continuity_pressure + 0.06)
        return AwakeDraft(
            user_turn=user_turn,
            user_fragment=user_fragment,
            user_traces=user_traces,
            activation=activation,
            response_hint=response_hint,
        )

    def commit_awake(
        self,
        draft: AwakeDraft,
        session_id: str,
        response_text: str,
        aurora_move: AuroraMove,
        now_ts: float,
    ) -> TurnCommit:
        aurora_turn = InteractionTurn(
            turn_id=f"turn_{uuid4().hex[:12]}",
            relation_id=draft.user_turn.relation_id,
            session_id=session_id,
            speaker=Speaker.AURORA,
            text=response_text,
            created_at=now_ts,
        )
        aurora_fragment = self.memory.create_fragment(
            relation_id=draft.user_turn.relation_id,
            turn_id=aurora_turn.turn_id,
            surface=response_text[:160],
            tags=extract_tags(response_text),
            vividness=0.42,
            salience=0.44,
            unresolvedness=0.28 if aurora_move == "withhold" else 0.18,
            now_ts=now_ts,
        )
        aurora_traces = tuple(
            self.memory.create_trace(
                relation_id=draft.user_turn.relation_id,
                fragment_id=aurora_fragment.fragment_id,
                channel=channel,
                intensity=intensity,
                now_ts=now_ts,
            )
            for channel, intensity in self._traces_from_move(aurora_move)
        )
        self.memory.link_fragments(
            draft.user_fragment.fragment_id,
            aurora_fragment.fragment_id,
            kind=self._edge_kind_from_move(aurora_move),
            weight=0.72,
            evidence=(draft.user_turn.turn_id, aurora_turn.turn_id),
            now_ts=now_ts,
        )
        self.relations.record_exchange(
            relation_id=draft.user_turn.relation_id,
            user_turn_id=draft.user_turn.turn_id,
            aurora_turn_id=aurora_turn.turn_id,
            user_channels=tuple(trace.channel for trace in draft.user_traces),
            aurora_move=aurora_move,
            summary=f"{draft.user_turn.text[:70]} -> {response_text[:70]}",
            now_ts=now_ts,
        )
        self._drift_being_from_commit(draft=draft, aurora_move=aurora_move)
        return TurnCommit(
            user_turn=draft.user_turn,
            aurora_turn=aurora_turn,
            user_fragment=draft.user_fragment,
            aurora_fragment=aurora_fragment,
            user_traces=draft.user_traces,
            aurora_traces=aurora_traces,
            response_text=response_text,
            aurora_move=aurora_move,
        )

    def doze(self, now_ts: float) -> None:
        self.being.phase = Phase.DOZE
        self.memory.decay_for_doze(now_ts=now_ts)
        self.being.sleep_pressure = min(1.0, self.being.sleep_pressure + 0.10)
        self.being.coherence_pressure = min(1.0, self.being.coherence_pressure + 0.08)

    def sleep(self, now_ts: float):
        self.being.phase = Phase.SLEEP
        result = self.memory.reweave(relation_states=self.relations.states, now_ts=now_ts)
        self.being.drift(result.self_drift, result.world_drift)
        self.being.coherence_pressure = max(0.0, self.being.coherence_pressure - result.coherence_shift)
        self.being.boundary_tension = min(1.0, max(0.0, self.being.boundary_tension + result.tension_shift - 0.05))
        self.being.sleep_pressure = max(0.0, self.being.sleep_pressure - 0.45)
        self.being.recent_chapter_bias = result.chapter_ids[-4:]
        return result

    def _hint_from_activation(self, text: str, activation: ActivationBundle) -> str:
        channels = ",".join(channel.value for channel in activation.dominant_channels) or "quiet"
        chapter_titles = ", ".join(chapter.title for chapter in activation.chapters[-2:]) or "none"
        return (
            f"dominant_channels={channels}; boundary_required={activation.boundary_required}; "
            f"recent_chapters={chapter_titles}; input_focus={text[:48]}"
        )

    def _infer_traces(self, text: str) -> list[tuple[TraceChannel, float]]:
        lower = text.lower()
        traces: list[tuple[TraceChannel, float]] = [(TraceChannel.COHERENCE, 0.25)]
        if any(token in text for token in ("谢谢", "信任", "理解", "陪", "在乎")) or any(token in lower for token in ("thanks", "trust", "understand")):
            traces.append((TraceChannel.WARMTH, 0.62))
            traces.append((TraceChannel.RECOGNITION, 0.48))
        if any(token in text for token in ("不", "害怕", "受伤", "失望", "痛")) or any(token in lower for token in ("hurt", "afraid", "disappointed")):
            traces.append((TraceChannel.HURT, 0.58))
        if any(token in text for token in ("边界", "不要", "停", "拒绝")) or any(token in lower for token in ("boundary", "stop", "no")):
            traces.append((TraceChannel.BOUNDARY, 0.72))
        if any(token in text for token in ("想知道", "为什么", "如何", "能否")) or any(token in lower for token in ("why", "how", "could")):
            traces.append((TraceChannel.CURIOSITY, 0.46))
        if any(token in text for token in ("修复", "弥补", "对不起", "和好")) or any(token in lower for token in ("repair", "sorry", "reconcile")):
            traces.append((TraceChannel.REPAIR, 0.60))
        if any(token in text for token in ("远", "冷", "疏离")) or any(token in lower for token in ("distant", "cold", "far")):
            traces.append((TraceChannel.DISTANCE, 0.50))
        return traces

    def _traces_from_move(self, aurora_move: AuroraMove) -> list[tuple[TraceChannel, float]]:
        if aurora_move == "boundary":
            return [(TraceChannel.BOUNDARY, 0.70), (TraceChannel.COHERENCE, 0.42)]
        if aurora_move == "repair":
            return [(TraceChannel.REPAIR, 0.66), (TraceChannel.WARMTH, 0.32)]
        if aurora_move == "withhold":
            return [(TraceChannel.DISTANCE, 0.48), (TraceChannel.COHERENCE, 0.32)]
        if aurora_move == "silence":
            return [(TraceChannel.DISTANCE, 0.38)]
        return [(TraceChannel.WARMTH, 0.28), (TraceChannel.RECOGNITION, 0.34)]

    def _edge_kind_from_move(self, aurora_move: AuroraMove):
        from core_models import AssocKind
        if aurora_move == "boundary":
            return AssocKind.BOUNDARY
        if aurora_move == "repair":
            return AssocKind.REPAIR
        if aurora_move == "withhold":
            return AssocKind.CONTRAST
        return AssocKind.RELATION

    def _estimate_unresolvedness(self, text: str) -> float:
        score = 0.35
        if any(token in text for token in ("为什么", "如何", "以后", "还会", "怎么办")):
            score += 0.20
        if any(token in text for token in ("?", "？")):
            score += 0.12
        if any(token in text for token in ("害怕", "失去", "矛盾", "边界", "不确定")):
            score += 0.18
        return min(1.0, score)

    def _drift_being_from_commit(self, draft: AwakeDraft, aurora_move: AuroraMove) -> None:
        channels = {trace.channel for trace in draft.user_traces}
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
            self.being.boundary_tension = min(1.0, self.being.boundary_tension + 0.08)
            world_updates["risk"] += 0.03
        if aurora_move in {"approach", "repair"}:
            self_updates["openness"] += 0.03
            self_updates["agency"] += 0.02
            world_updates["stability"] += 0.03
        if aurora_move == "withhold":
            self_updates["agency"] += 0.01
        self.being.drift(self_updates, world_updates)
