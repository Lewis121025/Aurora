from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .domain import (
    BeingState,
    Fragment,
    InteractionTurn,
    RelationMove,
    RelationState,
    TraceResidue,
)


class Clock(Protocol):
    def now(self) -> float:
        ...


@dataclass(frozen=True, slots=True)
class ImprintResult:
    fragments: tuple[Fragment, ...]
    traces: tuple[TraceResidue, ...]
    user_move: RelationMove


@dataclass(frozen=True, slots=True)
class ResponsePlan:
    surface: str
    aurora_move: RelationMove
    touched_fragment_ids: tuple[str, ...]


class Imprinter(Protocol):
    def imprint_user_turn(
        self,
        turn: InteractionTurn,
        relation: RelationState,
        being: BeingState,
    ) -> ImprintResult:
        ...

    def imprint_aurora_turn(
        self,
        turn: InteractionTurn,
        relation: RelationState,
        being: BeingState,
    ) -> ImprintResult:
        ...


class Responder(Protocol):
    def plan_response(
        self,
        user_turn: InteractionTurn,
        relation: RelationState,
        being: BeingState,
        recent_fragments: tuple[Fragment, ...],
        active_chapters: tuple[str, ...],
    ) -> ResponsePlan:
        ...
