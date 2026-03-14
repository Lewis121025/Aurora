from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from aurora.runtime.models import Tone
from aurora.runtime.models import AssociationDelta, Fragment, InteractionTurn, Trace


@dataclass(slots=True)
class MemoryStore:
    fragments: list[Fragment] = field(default_factory=list)
    traces: list[Trace] = field(default_factory=list)
    associations: list[AssociationDelta] = field(default_factory=list)
    fragment_salience: dict[str, float] = field(default_factory=dict)
    fragment_narrative_weight: dict[str, float] = field(default_factory=dict)
    sleep_cycles: int = 0
    last_reweave_delta: float = 0.0

    def remember_turn(
        self,
        turn: InteractionTurn,
        touch_modes: tuple[str, ...],
        created_at: float,
    ) -> tuple[Fragment, tuple[Trace, ...], tuple[AssociationDelta, ...]]:
        fragment = Fragment(
            fragment_id=f"frag_{uuid4().hex[:10]}",
            turn_id=turn.turn_id,
            text=turn.text.strip(),
            created_at=created_at,
        )
        self.fragments.append(fragment)
        initial_salience = self._initial_salience(touch_modes)
        self.fragment_salience[fragment.fragment_id] = initial_salience
        self.fragment_narrative_weight[fragment.fragment_id] = initial_salience

        traces = tuple(
            Trace(
                trace_id=f"tr_{uuid4().hex[:10]}",
                turn_id=turn.turn_id,
                mode=mode,
                intensity=self._trace_intensity(mode),
                created_at=created_at,
            )
            for mode in touch_modes
        )
        self.traces.extend(traces)

        association_items: list[AssociationDelta] = []
        if len(self.fragments) > 1:
            previous = self.fragments[-2]
            association_items.append(
                AssociationDelta(
                    association_id=f"as_{uuid4().hex[:10]}",
                    source_fragment_id=previous.fragment_id,
                    target_fragment_id=fragment.fragment_id,
                    weight=self._association_weight(touch_modes),
                    created_at=created_at,
                )
            )
        self.associations.extend(association_items)
        return fragment, traces, tuple(association_items)

    def fragment_by_id(self, fragment_id: str) -> Fragment | None:
        for fragment in self.fragments:
            if fragment.fragment_id == fragment_id:
                return fragment
        return None

    def trace_intensity_for_turn(self, turn_id: str) -> float:
        values = [trace.intensity for trace in self.traces if trace.turn_id == turn_id]
        if not values:
            return 0.0
        return sum(values) / len(values)

    def recent_activation(self, limit: int = 6) -> float:
        if not self.traces:
            return 0.0
        tail = self.traces[-limit:]
        return sum(trace.intensity for trace in tail) / len(tail)

    def narrative_weight_for_fragment(self, fragment_id: str) -> float:
        return self.fragment_narrative_weight.get(fragment_id, 0.0)

    def narrative_pressure(self, limit: int = 6) -> float:
        if not self.fragments:
            return 0.0
        tail = self.fragments[-limit:]
        values = [
            self.fragment_narrative_weight.get(fragment.fragment_id, 0.0) for fragment in tail
        ]
        if not values:
            return 0.0
        return sum(values) / len(values)

    def relation_alignment_for_turn(self, turn_id: str, relation_tone: Tone) -> float:
        has_warmth = any(
            trace.turn_id == turn_id and trace.mode == "warmth" for trace in self.traces
        )
        has_hurt = any(trace.turn_id == turn_id and trace.mode == "hurt" for trace in self.traces)
        has_boundary = any(
            trace.turn_id == turn_id and trace.mode == "boundary" for trace in self.traces
        )

        if relation_tone == "warm":
            return 1.0 if has_warmth else 0.35
        if relation_tone == "cold":
            return 1.0 if has_hurt else 0.30
        if relation_tone == "boundary":
            return 1.0 if has_boundary else 0.30
        return 0.5

    def association_strength_for_fragment(self, fragment_id: str) -> float:
        linked = [
            association.weight
            for association in self.associations
            if association.source_fragment_id == fragment_id
            or association.target_fragment_id == fragment_id
        ]
        if not linked:
            return 0.0
        return max(linked)

    @staticmethod
    def _trace_intensity(mode: str) -> float:
        if mode == "ambient":
            return 0.25
        if mode in {"warmth", "hurt"}:
            return 0.75
        if mode == "boundary":
            return 0.65
        return 0.50

    @staticmethod
    def _association_weight(touch_modes: tuple[str, ...]) -> float:
        if "hurt" in touch_modes or "warmth" in touch_modes:
            return 0.85
        if "insight" in touch_modes:
            return 0.65
        return 0.40

    @staticmethod
    def _initial_salience(touch_modes: tuple[str, ...]) -> float:
        if "hurt" in touch_modes or "warmth" in touch_modes:
            return 0.75
        if "insight" in touch_modes or "boundary" in touch_modes:
            return 0.65
        return 0.40
