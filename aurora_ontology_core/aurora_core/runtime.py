from __future__ import annotations

from dataclasses import dataclass
from time import time

from .being import BeingState
from .events import InteractionTurn, TouchSignal, make_id
from .expression import ExpressionPlanner, ResponseAct
from .lifecycle import DozeResult, LifecycleController
from .memory import Fragment, MemoryGraph
from .relation import RelationMoment, RelationSystem
from .reweave import NarrativeReweaver, ReweaveResult
from .schema import AuroraMoveKind, OtherMoveKind


@dataclass(frozen=True, slots=True)
class ExchangeResult:
    user_turn: InteractionTurn
    aurora_turn: InteractionTurn
    user_fragment: Fragment
    aurora_fragment: Fragment
    activated_fragments: tuple[Fragment, ...]
    response_act: ResponseAct
    relation_moment: RelationMoment
    current_phase: str


class AuroraCore:
    """A clean-slate ontology core.

    No profile-driven user image.
    No string-concatenation memory reweave.
    Expression is a downstream consequence of object states.
    """

    def __init__(self) -> None:
        self.being = BeingState()
        self.memory = MemoryGraph()
        self.relations = RelationSystem()
        self.reweaver = NarrativeReweaver()
        self.lifecycle = LifecycleController(self.reweaver)
        self.expression = ExpressionPlanner()

    def receive_user_turn(
        self,
        *,
        relation_id: str,
        session_id: str,
        text: str,
        other_move: OtherMoveKind = "share",
        signal: TouchSignal | None = None,
        now_ts: float | None = None,
    ) -> ExchangeResult:
        now_ts = time() if now_ts is None else now_ts
        signal = signal or self._infer_signal(text)
        self.lifecycle.enter_phase(self.being, "awake", now_ts)
        self.being.last_awake_at = now_ts

        user_turn = InteractionTurn(
            turn_id=make_id("turn"),
            relation_id=relation_id,
            session_id=session_id,
            speaker="user",
            text=text,
            created_at=now_ts,
        )
        user_fragment = self.memory.add_fragment(
            relation_id=relation_id,
            speaker="user",
            surface=text[:180],
            created_at=now_ts,
            turn_id=user_turn.turn_id,
            touch_signature=signal,
            tags=("awake_input",),
        )
        activated = self.memory.activate_for_relation(relation_id, signal)
        relation_state = self.relations.get(relation_id)
        act = self.expression.plan(
            being=self.being,
            relation=relation_state,
            signal=signal,
            activated_fragments=activated,
        )
        response_text = self.expression.render_placeholder(act, self.memory)
        aurora_turn = InteractionTurn(
            turn_id=make_id("turn"),
            relation_id=relation_id,
            session_id=session_id,
            speaker="aurora",
            text=response_text,
            created_at=now_ts,
            tags=(act.aurora_move, act.tone),
        )
        aurora_signal = self._signal_for_aurora_move(act.aurora_move)
        aurora_fragment = self.memory.add_fragment(
            relation_id=relation_id,
            speaker="aurora",
            surface=response_text[:180],
            created_at=now_ts,
            turn_id=aurora_turn.turn_id,
            touch_signature=aurora_signal,
            tags=("aurora_response", act.aurora_move),
        )
        self.memory.connect(
            src_fragment_id=user_fragment.fragment_id,
            dst_fragment_id=aurora_fragment.fragment_id,
            kind="causal_guess",
            weight=0.35,
            evidence="awake_exchange",
            now_ts=now_ts,
        )
        moment = self.relations.record_exchange(
            relation_id=relation_id,
            user_turn_id=user_turn.turn_id,
            aurora_turn_id=aurora_turn.turn_id,
            other_move=other_move,
            aurora_move=act.aurora_move,
            effect_signature=signal,
            boundary_event=act.aurora_move == "boundary",
            created_at=now_ts,
            note=act.proposition,
        )
        self._update_being_from_exchange(signal, act.aurora_move, activated)
        desired = self.lifecycle.desired_phase(self.being)
        if desired != "awake":
            self.lifecycle.enter_phase(self.being, desired, now_ts)
        return ExchangeResult(
            user_turn=user_turn,
            aurora_turn=aurora_turn,
            user_fragment=user_fragment,
            aurora_fragment=aurora_fragment,
            activated_fragments=tuple(activated),
            response_act=act,
            relation_moment=moment,
            current_phase=self.being.phase,
        )

    def doze_once(self, now_ts: float | None = None) -> DozeResult:
        now_ts = time() if now_ts is None else now_ts
        self.lifecycle.enter_phase(self.being, "doze", now_ts)
        result = self.lifecycle.doze(self.being, self.memory, now_ts)
        desired = self.lifecycle.desired_phase(self.being)
        if desired == "sleep":
            self.lifecycle.enter_phase(self.being, "sleep", now_ts)
        return result

    def sleep_once(self, now_ts: float | None = None) -> list[ReweaveResult]:
        now_ts = time() if now_ts is None else now_ts
        self.lifecycle.enter_phase(self.being, "sleep", now_ts)
        results = self.lifecycle.sleep(self.being, self.memory, self.relations, now_ts)
        next_phase = self.lifecycle.desired_phase(self.being)
        self.lifecycle.enter_phase(self.being, next_phase, now_ts)
        return results

    def tick_idle(self, now_ts: float | None = None) -> DozeResult | list[ReweaveResult] | None:
        now_ts = time() if now_ts is None else now_ts
        desired = self.lifecycle.desired_phase(self.being)
        if desired == "doze":
            return self.doze_once(now_ts)
        if desired == "sleep":
            return self.sleep_once(now_ts)
        return None

    def _update_being_from_exchange(
        self,
        signal: TouchSignal,
        aurora_move: AuroraMoveKind,
        activated: list[Fragment],
    ) -> None:
        self.being.sleep_pressure += 0.12 + 0.08 * signal.total_intensity()
        self.being.continuity_pressure += 0.05 + 0.04 * len(activated)
        self.being.coherence_pressure += 0.08 * signal.weights.get("hurt", 0.0)
        self.being.boundary_tension += 0.14 * signal.weights.get("boundary", 0.0)
        self.being.approach_drive += 0.10 * signal.weights.get("warmth", 0.0)
        self.being.approach_drive += 0.08 * signal.weights.get("recognition", 0.0)
        self.being.withdrawal_drive += 0.10 * signal.weights.get("distance", 0.0)
        self.being.withdrawal_drive += 0.08 if aurora_move == "boundary" else 0.0
        self.being.withdrawal_drive -= 0.06 if aurora_move == "approach" else 0.0
        self.being.active_chapter_ids = tuple(
            chapter_id
            for fragment in activated[:2]
            for chapter_id in fragment.chapter_ids
        )
        self.being.clamp()

    def _infer_signal(self, text: str) -> TouchSignal:
        lowered = text.lower()
        weights: dict[str, float] = {
            "warmth": 0.0,
            "hurt": 0.0,
            "recognition": 0.0,
            "distance": 0.0,
            "curiosity": 0.0,
            "boundary": 0.0,
            "repair": 0.0,
            "wonder": 0.0,
        }
        warmth_words = ["thank", "care", "warm", "喜欢", "信任", "谢谢"]
        hurt_words = ["hurt", "pain", "broken", "受伤", "痛", "失望"]
        recognition_words = ["remember", "see you", "理解", "记得", "看见"]
        distance_words = ["leave", "far", "cold", "疏远", "远离", "冷"]
        curiosity_words = ["why", "how", "wonder", "好奇", "为什么", "如何"]
        boundary_words = ["must", "force", "cross", "边界", "逼", "不能"]
        repair_words = ["sorry", "repair", "道歉", "修复", "补救"]
        wonder_words = ["dream", "night", "sleep", "梦", "睡", "星"]

        for word in warmth_words:
            if word in lowered:
                weights["warmth"] += 0.25
        for word in hurt_words:
            if word in lowered:
                weights["hurt"] += 0.30
        for word in recognition_words:
            if word in lowered:
                weights["recognition"] += 0.30
        for word in distance_words:
            if word in lowered:
                weights["distance"] += 0.20
        for word in curiosity_words:
            if word in lowered:
                weights["curiosity"] += 0.20
        for word in boundary_words:
            if word in lowered:
                weights["boundary"] += 0.35
        for word in repair_words:
            if word in lowered:
                weights["repair"] += 0.30
        for word in wonder_words:
            if word in lowered:
                weights["wonder"] += 0.20

        if not any(value > 0.0 for value in weights.values()):
            weights["curiosity"] = 0.20
            weights["recognition"] = 0.15
        capped = {key: min(value, 1.0) for key, value in weights.items() if value > 0.0}
        return TouchSignal(weights=capped, note="lexical-demo-interpreter")

    def _signal_for_aurora_move(self, move: AuroraMoveKind) -> TouchSignal:
        if move == "boundary":
            return TouchSignal(weights={"boundary": 0.70, "recognition": 0.20}, note="aurora-boundary")
        if move == "repair":
            return TouchSignal(weights={"repair": 0.65, "warmth": 0.25}, note="aurora-repair")
        if move == "approach":
            return TouchSignal(weights={"warmth": 0.55, "recognition": 0.30}, note="aurora-approach")
        if move == "withdraw":
            return TouchSignal(weights={"distance": 0.55, "boundary": 0.20}, note="aurora-withdraw")
        if move == "silence":
            return TouchSignal(weights={"wonder": 0.40, "distance": 0.20}, note="aurora-silence")
        return TouchSignal(weights={"recognition": 0.45, "curiosity": 0.20}, note="aurora-witness")
