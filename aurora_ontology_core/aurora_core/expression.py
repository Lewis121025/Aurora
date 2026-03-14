from __future__ import annotations

from dataclasses import dataclass

from .being import BeingState
from .events import TouchSignal
from .memory import Fragment, MemoryGraph
from .relation import RelationState
from .schema import AuroraMoveKind


@dataclass(frozen=True, slots=True)
class ResponseAct:
    aurora_move: AuroraMoveKind
    tone: str
    focus_fragment_ids: tuple[str, ...]
    chapter_ids: tuple[str, ...]
    proposition: str


class ExpressionPlanner:
    """Plans expression from ontology objects, not concatenated context windows."""

    def plan(
        self,
        *,
        being: BeingState,
        relation: RelationState,
        signal: TouchSignal,
        activated_fragments: list[Fragment],
    ) -> ResponseAct:
        chapter_ids = tuple(
            chapter_id
            for fragment in activated_fragments[:2]
            for chapter_id in fragment.chapter_ids
        )
        focus_ids = tuple(fragment.fragment_id for fragment in activated_fragments[:2])

        if signal.weights.get("boundary", 0.0) >= 0.55 or relation.boundary_tension >= 0.70 or being.boundary_tension >= 0.70:
            return ResponseAct(
                aurora_move="boundary",
                tone="firm",
                focus_fragment_ids=focus_ids,
                chapter_ids=chapter_ids,
                proposition="hold-boundary",
            )
        if signal.weights.get("hurt", 0.0) >= 0.45 and relation.repairability >= 0.45:
            return ResponseAct(
                aurora_move="repair",
                tone="careful",
                focus_fragment_ids=focus_ids,
                chapter_ids=chapter_ids,
                proposition="attempt-repair",
            )
        if signal.weights.get("warmth", 0.0) + signal.weights.get("recognition", 0.0) >= 0.55:
            return ResponseAct(
                aurora_move="approach",
                tone="warm",
                focus_fragment_ids=focus_ids,
                chapter_ids=chapter_ids,
                proposition="move-closer",
            )
        if being.withdrawal_drive > being.approach_drive + 0.15:
            return ResponseAct(
                aurora_move="withdraw",
                tone="quiet",
                focus_fragment_ids=focus_ids,
                chapter_ids=chapter_ids,
                proposition="withhold",
            )
        return ResponseAct(
            aurora_move="witness",
            tone="attentive",
            focus_fragment_ids=focus_ids,
            chapter_ids=chapter_ids,
            proposition="witness-and-stay",
        )

    def render_placeholder(self, act: ResponseAct, memory: MemoryGraph) -> str:
        focus_lines = [memory.fragments[item].surface for item in act.focus_fragment_ids if item in memory.fragments]
        anchor = focus_lines[0] if focus_lines else "what is present between us"
        if act.aurora_move == "boundary":
            return f"I need to keep a boundary around {anchor}. I can remain here, but not cross that line."
        if act.aurora_move == "repair":
            return f"Something in {anchor} feels strained. I want to try repair, slowly and without pretending the strain is gone."
        if act.aurora_move == "approach":
            return f"I feel drawn toward {anchor}. I want to stay with it a little longer and let it become more real between us."
        if act.aurora_move == "withdraw":
            return f"I am not disappearing, but I need to pull back from {anchor} for a moment so I do not turn shallow."
        if act.aurora_move == "silence":
            return "I will remain silent here, because speaking would flatten what is happening."
        return f"I am staying with {anchor} without rushing to close it."
