from __future__ import annotations

from dataclasses import dataclass, replace

from aurora.runtime.contracts import clamp


@dataclass(frozen=True, slots=True)
class Fragment:
    fragment_id: str
    relation_id: str
    turn_id: str | None
    surface: str
    tags: tuple[str, ...]
    vividness: float
    salience: float
    unresolvedness: float
    thread_ids: tuple[str, ...]
    knot_ids: tuple[str, ...]
    created_at: float
    last_touched_at: float
    activation_count: int = 0

    def touched(
        self,
        at: float,
        delta_salience: float = 0.08,
        delta_unresolved: float = 0.0,
    ) -> "Fragment":
        return replace(
            self,
            salience=clamp(self.salience + delta_salience),
            unresolvedness=clamp(self.unresolvedness + delta_unresolved),
            last_touched_at=at,
            activation_count=self.activation_count + 1,
        )
