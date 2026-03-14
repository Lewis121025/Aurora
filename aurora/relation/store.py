from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from aurora.runtime.models import InteractionTurn, RelationMoment, Tone


@dataclass(slots=True)
class RelationStore:
    moments: list[RelationMoment] = field(default_factory=list)

    def record_turn(
        self,
        turn: InteractionTurn,
        touch_modes: tuple[str, ...],
        created_at: float,
    ) -> RelationMoment:
        tone = self._tone_from_touch(turn.text, touch_modes)
        moment = RelationMoment(
            moment_id=f"rel_{uuid4().hex[:10]}",
            session_id=turn.session_id,
            turn_id=turn.turn_id,
            tone=tone,
            summary=turn.text[:120],
            created_at=created_at,
        )
        self.moments.append(moment)
        return moment

    def current_tone(self) -> Tone:
        scores = self._tone_scores()
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not ranked or ranked[0][1] <= 0.0:
            return "neutral"
        return ranked[0][0]

    def tone_strength(self, tone: Tone) -> float:
        scores = self._tone_scores()
        total = sum(scores.values())
        if total <= 0.0:
            return 0.0
        return scores[tone] / total

    def _tone_scores(self) -> dict[Tone, float]:
        if not self.moments:
            return {"warm": 0.0, "neutral": 1.0, "cold": 0.0, "boundary": 0.0}

        scores: dict[Tone, float] = {"warm": 0.0, "neutral": 0.0, "cold": 0.0, "boundary": 0.0}
        for index, moment in enumerate(reversed(self.moments), start=1):
            decay = 1.0 / float(index)
            scores[moment.tone] += decay
        return scores

    @staticmethod
    def _tone_from_touch(text: str, touch_modes: tuple[str, ...]) -> Tone:
        lowered = text.lower()
        if "boundary" in touch_modes:
            return "boundary"
        if "hurt" in touch_modes or any(word in lowered for word in ("idiot", "stupid", "滚")):
            return "cold"
        if "warmth" in touch_modes:
            return "warm"
        return "neutral"
