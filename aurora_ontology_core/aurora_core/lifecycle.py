from __future__ import annotations

from dataclasses import dataclass

from .being import BeingState
from .memory import MemoryGraph
from .relation import RelationSystem
from .reweave import NarrativeReweaver, ReweaveResult
from .schema import Phase


@dataclass(frozen=True, slots=True)
class DozeResult:
    refreshed_fragment_ids: tuple[str, ...]
    coherence_delta: float
    sleep_delta: float


class LifecycleController:
    def __init__(self, reweaver: NarrativeReweaver) -> None:
        self.reweaver = reweaver

    def desired_phase(self, being: BeingState) -> Phase:
        if being.sleep_pressure >= 0.70 or being.coherence_pressure >= 0.75:
            return "sleep"
        if being.sleep_pressure >= 0.40 or being.continuity_pressure >= 0.45:
            return "doze"
        return "awake"

    def enter_phase(self, being: BeingState, phase: Phase, now_ts: float) -> None:
        if being.phase != phase:
            being.phase = phase
            being.last_transition_at = now_ts
            being.note(f"phase->{phase}@{now_ts:.0f}")

    def doze(self, being: BeingState, memory: MemoryGraph, now_ts: float) -> DozeResult:
        candidates = memory.select_reweave_candidates(limit=8)
        refreshed: list[str] = []
        for fragment in candidates[:4]:
            memory.touch_fragment(fragment.fragment_id, salience_delta=0.03, unresolved_delta=-0.02)
            refreshed.append(fragment.fragment_id)
        being.coherence_pressure = max(0.0, being.coherence_pressure - 0.05)
        being.sleep_pressure = min(1.0, being.sleep_pressure + 0.10)
        being.continuity_pressure = max(0.0, being.continuity_pressure - 0.03)
        being.clamp()
        return DozeResult(
            refreshed_fragment_ids=tuple(refreshed),
            coherence_delta=-0.05,
            sleep_delta=0.10,
        )

    def sleep(self, being: BeingState, memory: MemoryGraph, relations: RelationSystem, now_ts: float) -> list[ReweaveResult]:
        results = self.reweaver.reweave_all(memory, now_ts=now_ts)
        accumulated_tension = 0.0
        accumulated_bias = 0.0
        chapter_ids: list[str] = []
        for result in results:
            relations.absorb_reweave(
                result.relation_id,
                chapter_ids=result.chapter_ids,
                relation_bias=result.relation_bias,
                tension_shift=result.tension_shift,
            )
            accumulated_tension += result.tension_shift
            accumulated_bias += result.relation_bias
            chapter_ids.extend(result.chapter_ids)

        being.coherence_pressure = max(0.0, being.coherence_pressure - 0.25)
        being.sleep_pressure = max(0.0, being.sleep_pressure - 0.45)
        being.continuity_pressure = max(0.0, being.continuity_pressure - 0.20)
        being.boundary_tension = min(max(being.boundary_tension + accumulated_tension * 0.15, 0.0), 1.0)
        being.approach_drive = min(max(being.approach_drive + accumulated_bias * 0.10, 0.0), 1.0)
        being.active_chapter_ids = tuple(chapter_ids[-6:])
        being.clamp()
        return results
