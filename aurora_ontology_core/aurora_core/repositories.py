from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from itertools import combinations
from typing import Iterable
from uuid import uuid4

from .domain import (
    AssocKind,
    AssociationEdge,
    Chapter,
    ChapterStatus,
    Fragment,
    RelationMoment,
    RelationState,
    TraceChannel,
    TraceResidue,
)


class InMemoryMemoryGraph:
    """对象本体层的最小内存图。

    注意：这里的 dict 只是存储实现细节，系统语义仍然由 Fragment / Trace /
    Association / Chapter 这些对象定义，而不是由扁平 key-value 定义。
    """

    def __init__(self) -> None:
        self.fragments: dict[str, Fragment] = {}
        self.traces: dict[str, TraceResidue] = {}
        self.associations: dict[tuple[str, str, AssocKind], AssociationEdge] = {}
        self.chapters: dict[str, Chapter] = {}
        self.chapter_membership: dict[str, set[str]] = defaultdict(set)

    def add_fragments(self, fragments: Iterable[Fragment]) -> None:
        for fragment in fragments:
            self.fragments[fragment.fragment_id] = fragment

    def add_traces(self, traces: Iterable[TraceResidue]) -> None:
        for trace in traces:
            self.traces[trace.trace_id] = trace

    def touch_fragments(self, fragment_ids: Iterable[str], now_ts: float, boost: float = 0.12) -> None:
        for fragment_id in fragment_ids:
            if fragment_id not in self.fragments:
                continue
            current = self.fragments[fragment_id]
            self.fragments[fragment_id] = replace(
                current,
                activation=min(1.0, current.activation + boost),
                salience=min(1.0, current.salience + boost * 0.5),
                last_touched_at=now_ts,
            )

    def add_or_strengthen_association(
        self,
        src: str,
        dst: str,
        kind: AssocKind,
        weight_delta: float,
        now_ts: float,
    ) -> None:
        if src == dst:
            return
        key = tuple(sorted((src, dst))) + (kind,)
        existing = self.associations.get(key)
        if existing is None:
            self.associations[key] = AssociationEdge(
                edge_id=f"edge_{uuid4().hex[:10]}",
                src_fragment_id=key[0],
                dst_fragment_id=key[1],
                kind=kind,
                weight=max(0.0, min(1.0, weight_delta)),
                evidence_count=1,
                created_at=now_ts,
                last_touched_at=now_ts,
            )
            return
        self.associations[key] = replace(
            existing,
            weight=max(0.0, min(1.0, existing.weight + weight_delta)),
            evidence_count=existing.evidence_count + 1,
            last_touched_at=now_ts,
        )

    def link_exchange(self, fragment_ids: tuple[str, ...], now_ts: float) -> None:
        for left, right in combinations(fragment_ids, 2):
            left_fragment = self.fragments[left]
            right_fragment = self.fragments[right]
            overlap = set(left_fragment.touch_channels) & set(right_fragment.touch_channels)
            if overlap:
                self.add_or_strengthen_association(left, right, AssocKind.RESONANCE, 0.16, now_ts)
            else:
                self.add_or_strengthen_association(left, right, AssocKind.TEMPORAL, 0.08, now_ts)

    def recent_fragments_for_relation(
        self,
        relation_id: str,
        limit: int = 8,
        preferred_channels: tuple[TraceChannel, ...] = (),
    ) -> tuple[Fragment, ...]:
        channel_set = set(preferred_channels)
        candidates = [f for f in self.fragments.values() if f.relation_id == relation_id]

        def score(fragment: Fragment) -> float:
            base = fragment.activation * 0.40 + fragment.salience * 0.35 + fragment.unresolvedness * 0.15
            channel_bonus = 0.0
            if channel_set and channel_set.intersection(fragment.touch_channels):
                channel_bonus += 0.20
            chapter_bonus = 0.05 * sum(1 for members in self.chapter_membership.values() if fragment.fragment_id in members)
            return base + channel_bonus + chapter_bonus

        ranked = sorted(candidates, key=score, reverse=True)
        return tuple(ranked[:limit])

    def decay_for_doze(self, now_ts: float) -> None:
        for fragment_id, fragment in list(self.fragments.items()):
            self.fragments[fragment_id] = replace(
                fragment,
                activation=max(0.0, fragment.activation * 0.72),
                salience=max(0.05, fragment.salience * 0.98),
                last_touched_at=now_ts,
            )
        for trace_id, trace in list(self.traces.items()):
            self.traces[trace_id] = replace(
                trace,
                intensity=max(0.0, trace.intensity * 0.96),
                last_touched_at=now_ts,
            )

    def consolidate_recent_patterns(self, relation_id: str, now_ts: float, window: int = 6) -> None:
        recent = self.recent_fragments_for_relation(relation_id=relation_id, limit=window)
        for left, right in combinations(recent, 2):
            shared = set(left.touch_channels) & set(right.touch_channels)
            if shared:
                self.add_or_strengthen_association(
                    left.fragment_id,
                    right.fragment_id,
                    AssocKind.RESONANCE,
                    0.05 + 0.02 * len(shared),
                    now_ts,
                )

    def select_reweave_candidates(self, top_k: int = 24) -> tuple[Fragment, ...]:
        def score(fragment: Fragment) -> float:
            centrality = 0.0
            for edge in self.associations.values():
                if fragment.fragment_id in (edge.src_fragment_id, edge.dst_fragment_id):
                    centrality += edge.weight * 0.05
            chapter_penalty = 0.05 * sum(1 for members in self.chapter_membership.values() if fragment.fragment_id in members)
            return (
                fragment.salience * 0.35
                + fragment.unresolvedness * 0.30
                + fragment.activation * 0.20
                + centrality
                - chapter_penalty
            )

        ranked = sorted(self.fragments.values(), key=score, reverse=True)
        return tuple(ranked[:top_k])

    def fragments_by_ids(self, fragment_ids: Iterable[str]) -> tuple[Fragment, ...]:
        return tuple(self.fragments[fragment_id] for fragment_id in fragment_ids if fragment_id in self.fragments)

    def association_neighbors(self, fragment_id: str) -> tuple[AssociationEdge, ...]:
        return tuple(
            edge
            for edge in self.associations.values()
            if fragment_id in (edge.src_fragment_id, edge.dst_fragment_id)
        )

    def apply_chapter(self, chapter: Chapter, now_ts: float) -> tuple[int, int]:
        self.chapters[chapter.chapter_id] = chapter
        self.chapter_membership[chapter.chapter_id] = set(chapter.fragment_ids)
        created_edges = 0
        for fragment_id in chapter.fragment_ids:
            self.touch_fragments((fragment_id,), now_ts=now_ts, boost=0.06)
            fragment = self.fragments[fragment_id]
            self.fragments[fragment_id] = replace(
                fragment,
                unresolvedness=max(0.0, fragment.unresolvedness * 0.92),
                last_touched_at=now_ts,
            )
        for left, right in combinations(chapter.fragment_ids, 2):
            before = len(self.associations)
            self.add_or_strengthen_association(left, right, AssocKind.CHAPTER, 0.18, now_ts)
            after = len(self.associations)
            if after > before:
                created_edges += 1
        return 0, created_edges

    def chapter_ids_for_relation(self, relation_id: str) -> tuple[str, ...]:
        chapters = [chapter.chapter_id for chapter in self.chapters.values() if chapter.relation_id == relation_id]
        chapters.sort(key=lambda chapter_id: self.chapters[chapter_id].last_rewoven_at, reverse=True)
        return tuple(chapters)


class InMemoryRelationRepository:
    def __init__(self) -> None:
        self.states: dict[str, RelationState] = {}
        self.moments: dict[str, list[RelationMoment]] = defaultdict(list)

    def get(self, relation_id: str) -> RelationState:
        if relation_id not in self.states:
            self.states[relation_id] = RelationState(relation_id=relation_id)
        return self.states[relation_id]

    def record_moment(self, moment: RelationMoment) -> None:
        self.moments[moment.relation_id].append(moment)
        state = self.get(moment.relation_id)

        trust = state.trust + 0.08 * moment.resonance_score - 0.10 * moment.boundary_signal
        familiarity = state.familiarity + 0.06
        reciprocity = state.reciprocity + (0.08 if moment.aurora_move != moment.user_move else 0.05)
        boundary_tension = max(0.0, state.boundary_tension * 0.92 + moment.boundary_signal * 0.35)
        repairability = min(1.0, max(0.0, state.repairability + 0.05 * (1.0 - moment.boundary_signal)))
        motifs = tuple(sorted(set(state.motif_channels).union(moment.user_channels), key=lambda x: x.value))

        self.states[moment.relation_id] = replace(
            state,
            trust=max(-1.0, min(1.0, trust)),
            familiarity=min(1.0, familiarity),
            reciprocity=min(1.0, reciprocity),
            boundary_tension=min(1.0, boundary_tension),
            repairability=repairability,
            motif_channels=motifs,
            last_contact_at=moment.created_at,
        )

    def integrate_chapters(self, relation_id: str, chapter_ids: tuple[str, ...], relation_bias: float) -> None:
        state = self.get(relation_id)
        active = tuple(dict.fromkeys(chapter_ids + state.active_chapter_ids))[:6]
        self.states[relation_id] = replace(
            state,
            active_chapter_ids=active,
            trust=max(-1.0, min(1.0, state.trust + relation_bias * 0.10)),
            reciprocity=max(0.0, min(1.0, state.reciprocity + relation_bias * 0.05)),
        )
