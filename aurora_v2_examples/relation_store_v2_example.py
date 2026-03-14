from __future__ import annotations

from dataclasses import dataclass, field
from math import exp
from typing import Iterable
from uuid import uuid4

from models_v2_example import InteractionTurn, RelationMoment, RelationState, Tone, TraceChannel


DEFAULT_TOUCH_TO_TONE: dict[Tone, dict[str, float]] = {
    "warm": {
        "warmth": 1.0,
        "recognition": 0.9,
        "curiosity": 0.25,
        "repair": 0.65,
    },
    "neutral": {
        "curiosity": 0.30,
    },
    "cold": {
        "distance": 0.9,
        "hurt": 0.8,
    },
    "boundary": {
        "boundary": 1.0,
        "pressure": 0.85,
    },
}


@dataclass(slots=True)
class RelationAggregate:
    state: RelationState
    moments: list[RelationMoment] = field(default_factory=list)


@dataclass(slots=True)
class RelationStoreV2:
    """
    与当前 RelationStore 最大的区别：
    1. 按 relation_id 持续保存状态
    2. 记录 user + aurora 双向动作
    3. 支持可配置的 decay mode
    """
    relations: dict[str, RelationAggregate] = field(default_factory=dict)
    tone_decay_mode: str = "harmonic"
    touch_to_tone: dict[Tone, dict[str, float]] = field(default_factory=lambda: DEFAULT_TOUCH_TO_TONE.copy())

    def ensure_relation(self, relation_id: str) -> RelationAggregate:
        if relation_id not in self.relations:
            self.relations[relation_id] = RelationAggregate(state=RelationState(relation_id=relation_id))
        return self.relations[relation_id]

    def record_exchange(
        self,
        relation_id: str,
        session_id: str,
        user_turn: InteractionTurn,
        aurora_turn: InteractionTurn | None,
        touch_modes: tuple[str, ...],
        user_move: str = "share",
        aurora_move: str = "approach",
        effect_channels: tuple[TraceChannel, ...] = (),
        created_at: float = 0.0,
    ) -> RelationMoment:
        aggregate = self.ensure_relation(relation_id)
        tone = self._tone_from_touch(touch_modes)
        moment = RelationMoment(
            moment_id=f"rel_{uuid4().hex[:10]}",
            relation_id=relation_id,
            session_id=session_id,
            user_turn_id=user_turn.turn_id,
            aurora_turn_id=aurora_turn.turn_id if aurora_turn else None,
            tone=tone,
            user_move=user_move,          # 例如 share / pressure / repair
            aurora_move=aurora_move,      # 例如 approach / boundary / withhold / repair
            effect_channels=effect_channels,
            summary=self._summarize_exchange(user_turn, aurora_turn),
            created_at=created_at,
        )
        aggregate.moments.append(moment)
        self._update_state(aggregate.state, moment)
        return moment

    def current_tone(self, relation_id: str) -> Tone:
        aggregate = self.ensure_relation(relation_id)
        scores = self._tone_scores(aggregate.moments)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not ranked or ranked[0][1] <= 0.0:
            return "neutral"
        return ranked[0][0]

    def tone_strength(self, relation_id: str, tone: Tone) -> float:
        aggregate = self.ensure_relation(relation_id)
        scores = self._tone_scores(aggregate.moments)
        total = sum(scores.values())
        if total <= 0.0:
            return 0.0
        return scores[tone] / total

    def chapter_ids_for_relation(self, relation_id: str) -> tuple[str, ...]:
        aggregate = self.ensure_relation(relation_id)
        return tuple(sorted(aggregate.state.shared_chapter_ids))

    def _tone_scores(self, moments: Iterable[RelationMoment]) -> dict[Tone, float]:
        listed = list(moments)
        if not listed:
            return {"warm": 0.0, "neutral": 1.0, "cold": 0.0, "boundary": 0.0}

        scores: dict[Tone, float] = {"warm": 0.0, "neutral": 0.0, "cold": 0.0, "boundary": 0.0}
        for index, moment in enumerate(reversed(listed), start=1):
            decay = self._decay_at(index)
            scores[moment.tone] += decay
        return scores

    def _decay_at(self, index: int) -> float:
        if self.tone_decay_mode == "flat":
            return 1.0
        if self.tone_decay_mode == "exp":
            return exp(-0.45 * float(index - 1))
        # 默认与当前实现接近，但显式命名为 harmonic
        return 1.0 / float(index)

    def _update_state(self, state: RelationState, moment: RelationMoment) -> None:
        if moment.tone == "warm":
            state.trust = self._clamp(state.trust + 0.08, -1.0, 1.0)
            state.boundary_tension = self._clamp(state.boundary_tension - 0.04, 0.0, 1.0)
        elif moment.tone == "cold":
            state.trust = self._clamp(state.trust - 0.06, -1.0, 1.0)
            state.boundary_tension = self._clamp(state.boundary_tension + 0.07, 0.0, 1.0)
        elif moment.tone == "boundary":
            state.boundary_tension = self._clamp(state.boundary_tension + 0.15, 0.0, 1.0)
            state.repairability = self._clamp(state.repairability - 0.03, 0.0, 1.0)

        if moment.aurora_move == "repair" or moment.user_move == "repair":
            state.repairability = self._clamp(state.repairability + 0.06, 0.0, 1.0)

        if moment.aurora_move == "withhold":
            state.boundary_tension = self._clamp(state.boundary_tension + 0.05, 0.0, 1.0)

        state.current_tone = self.current_tone(state.relation_id)

    def _tone_from_touch(self, touch_modes: tuple[str, ...]) -> Tone:
        observed = set(touch_modes)
        scores: dict[Tone, float] = {"warm": 0.0, "neutral": 0.0, "cold": 0.0, "boundary": 0.0}
        for tone, mapping in self.touch_to_tone.items():
            for mode, weight in mapping.items():
                if mode in observed:
                    scores[tone] += weight

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not ranked or ranked[0][1] <= 0.0:
            return "neutral"
        return ranked[0][0]

    @staticmethod
    def _summarize_exchange(user_turn: InteractionTurn, aurora_turn: InteractionTurn | None) -> str:
        left = user_turn.text[:60].replace("\n", " ").strip()
        if aurora_turn is None:
            return f"U:{left}"
        right = aurora_turn.text[:60].replace("\n", " ").strip()
        return f"U:{left} | A:{right}"

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))
