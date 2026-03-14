from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from .schema import Speaker, TraceChannel


SignalWeights = dict[TraceChannel, float]


def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:10]}"


@dataclass(frozen=True, slots=True)
class InteractionTurn:
    turn_id: str
    relation_id: str
    session_id: str
    speaker: Speaker
    text: str
    created_at: float
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TouchSignal:
    weights: SignalWeights = field(default_factory=dict)
    note: str = ""

    def dominant_channels(self, threshold: float = 0.35) -> tuple[TraceChannel, ...]:
        dominant = [channel for channel, value in self.weights.items() if value >= threshold]
        if dominant:
            return tuple(sorted(dominant))
        if not self.weights:
            return ()
        max_channel = max(self.weights.items(), key=lambda item: item[1])[0]
        return (max_channel,)

    def total_intensity(self) -> float:
        return sum(self.weights.values())

    def overlap(self, other: TouchSignal) -> float:
        score = 0.0
        for channel, value in self.weights.items():
            score += min(value, other.weights.get(channel, 0.0))
        return score
