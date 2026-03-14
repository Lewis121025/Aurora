from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from itertools import combinations
import math
import re
from typing import Iterable
from uuid import uuid4

from aurora.runtime.models import (
    AssocKind,
    Association,
    Chapter,
    ChapterRole,
    Fragment,
    RelationState,
    ReweaveMutation,
    Trace,
    TraceChannel,
    clamp,
)


class MemoryStore:
    def __init__(self) -> None:
        self.fragments: dict[str, Fragment] = {}
        self.traces: dict[str, Trace] = {}
        self.associations: dict[str, Association] = {}
        self.chapters: dict[str, Chapter] = {}
        self.fragment_traces: dict[str, list[str]] = defaultdict(list)
        self.relation_fragments: dict[str, list[str]] = defaultdict(list)
        self.fragment_edges: dict[str, set[str]] = defaultdict(set)
        self.sleep_cycles = 0
        self.last_reweave_delta = 0.0

    def add_fragment(self, fragment: Fragment) -> None:
        self.fragments[fragment.fragment_id] = fragment
        if fragment.fragment_id not in self.relation_fragments[fragment.relation_id]:
            self.relation_fragments[fragment.relation_id].append(fragment.fragment_id)

    def add_trace(self, trace: Trace) -> None:
        self.traces[trace.trace_id] = trace
        if trace.trace_id not in self.fragment_traces[trace.fragment_id]:
            self.fragment_traces[trace.fragment_id].append(trace.trace_id)

    def add_association(self, edge: Association) -> None:
        self.associations[edge.edge_id] = edge
        self.fragment_edges[edge.src_fragment_id].add(edge.edge_id)
        self.fragment_edges[edge.dst_fragment_id].add(edge.edge_id)

    def add_chapter(self, chapter: Chapter) -> None:
        self.chapters[chapter.chapter_id] = chapter
        for fragment_id in chapter.fragment_ids:
            fragment = self.fragments[fragment_id]
            if chapter.chapter_id not in fragment.chapter_ids:
                self.fragments[fragment_id] = replace(
                    fragment,
                    chapter_ids=tuple(sorted(fragment.chapter_ids + (chapter.chapter_id,))),
                )

    def create_fragment(
        self,
        relation_id: str,
        turn_id: str | None,
        surface: str,
        vividness: float,
        salience: float,
        unresolvedness: float,
        now_ts: float,
    ) -> Fragment:
        fragment = Fragment(
            fragment_id=f"frag_{uuid4().hex[:12]}",
            relation_id=relation_id,
            turn_id=turn_id,
            surface=surface,
            tags=extract_tags(surface),
            vividness=clamp(vividness),
            salience=clamp(salience),
            unresolvedness=clamp(unresolvedness),
            chapter_ids=(),
            created_at=now_ts,
            last_touched_at=now_ts,
        )
        self.add_fragment(fragment)
        return fragment

    def create_trace(
        self,
        relation_id: str,
        fragment_id: str,
        channel: TraceChannel,
        intensity: float,
        now_ts: float,
        decay_rate: float = 0.03,
    ) -> Trace:
        trace = Trace(
            trace_id=f"trace_{uuid4().hex[:12]}",
            relation_id=relation_id,
            fragment_id=fragment_id,
            channel=channel,
            intensity=clamp(intensity),
            decay_rate=clamp(decay_rate),
            created_at=now_ts,
            last_touched_at=now_ts,
        )
        self.add_trace(trace)
        return trace

    def link_fragments(
        self,
        src_fragment_id: str,
        dst_fragment_id: str,
        kind: AssocKind,
        weight: float,
        evidence: Iterable[str],
        now_ts: float,
    ) -> Association:
        edge = Association(
            edge_id=f"edge_{uuid4().hex[:12]}",
            src_fragment_id=src_fragment_id,
            dst_fragment_id=dst_fragment_id,
            kind=kind,
            weight=clamp(weight),
            evidence=tuple(evidence),
            created_at=now_ts,
            last_touched_at=now_ts,
        )
        self.add_association(edge)
        return edge

    def traces_for_fragment(self, fragment_id: str) -> tuple[Trace, ...]:
        return tuple(
            self.traces[trace_id] for trace_id in self.fragment_traces.get(fragment_id, [])
        )

    def fragments_for_relation(self, relation_id: str) -> tuple[Fragment, ...]:
        return tuple(
            self.fragments[fragment_id]
            for fragment_id in self.relation_fragments.get(relation_id, [])
        )

    def chapters_for_relation(self, relation_id: str) -> tuple[Chapter, ...]:
        return tuple(
            chapter for chapter in self.chapters.values() if chapter.relation_id == relation_id
        )

    def touch_fragment(self, fragment_id: str, at: float, delta_salience: float = 0.08) -> None:
        self.fragments[fragment_id] = self.fragments[fragment_id].touched(
            at=at, delta_salience=delta_salience
        )

    def recent_recall(self, relation_id: str, limit: int = 8) -> tuple[Fragment, ...]:
        ranked = sorted(
            self.fragments_for_relation(relation_id),
            key=lambda item: (
                0.42 * item.salience
                + 0.25 * item.unresolvedness
                + 0.18 * min(item.activation_count / 4.0, 1.0)
                + 0.15 * self._novelty_bonus(item)
            ),
            reverse=True,
        )
        return tuple(ranked[:limit])

    def build_activation_channels(self, fragments: Iterable[Fragment]) -> tuple[TraceChannel, ...]:
        scores: dict[TraceChannel, float] = {}
        for fragment in fragments:
            for trace in self.traces_for_fragment(fragment.fragment_id):
                scores[trace.channel] = scores.get(trace.channel, 0.0) + trace.intensity
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return tuple(channel for channel, _ in ranked[:4])

    def decay_for_doze(self, now_ts: float) -> None:
        for fragment_id, fragment in list(self.fragments.items()):
            hours = max(0.0, (now_ts - fragment.last_touched_at) / 3600.0)
            drop = min(0.12, hours * 0.01)
            self.fragments[fragment_id] = replace(
                fragment, salience=clamp(fragment.salience - drop)
            )
        for trace_id, trace in list(self.traces.items()):
            hours = max(0.0, (now_ts - trace.last_touched_at) / 3600.0)
            next_intensity = clamp(trace.intensity - trace.decay_rate * min(1.0, hours / 12.0))
            self.traces[trace_id] = replace(trace, intensity=next_intensity)

    def reweave(self, relation_states: dict[str, RelationState], now_ts: float) -> ReweaveMutation:
        candidates = self._select_reweave_candidates()
        clusters = [cluster for cluster in self._simple_cluster(candidates) if len(cluster) >= 2]

        created_chapters: list[str] = []
        updated_fragments: set[str] = set()
        strengthened_edges: list[str] = []
        notes: list[str] = []
        self_drift = {"recognition": 0.0, "fragility": 0.0, "openness": 0.0, "agency": 0.0}
        world_drift = {"welcome": 0.0, "risk": 0.0, "mystery": 0.0, "stability": 0.0}
        coherence_shift = 0.0
        tension_shift = 0.0

        for cluster in clusters:
            relation_id = cluster[0].relation_id
            chapter = self._build_chapter(relation_id=relation_id, cluster=cluster, now_ts=now_ts)
            self.add_chapter(chapter)
            created_chapters.append(chapter.chapter_id)
            notes.append(f"{chapter.title}: {chapter.motif}")
            if relation_id in relation_states:
                relation_states[relation_id].shared_chapters.add(chapter.chapter_id)
            coherence_shift += chapter.coherence * 0.08
            tension_shift += chapter.tension * 0.06

            if chapter.tension > 0.55:
                self_drift["fragility"] += 0.05
                world_drift["risk"] += 0.04
            else:
                self_drift["recognition"] += 0.04
                world_drift["welcome"] += 0.03
            if "repair" in chapter.motif:
                self_drift["openness"] += 0.05
                world_drift["stability"] += 0.04
            else:
                world_drift["mystery"] += 0.02

            for fragment in cluster:
                updated_fragments.add(fragment.fragment_id)
                self.fragments[fragment.fragment_id] = replace(
                    self.fragments[fragment.fragment_id],
                    salience=clamp(fragment.salience + 0.08),
                    unresolvedness=clamp(fragment.unresolvedness - 0.10),
                    last_touched_at=now_ts,
                )

            for left, right in combinations(cluster, 2):
                edge = self.link_fragments(
                    src_fragment_id=left.fragment_id,
                    dst_fragment_id=right.fragment_id,
                    kind=AssocKind.CHAPTER,
                    weight=0.55 + 0.35 * self._affinity(left, right),
                    evidence=(chapter.chapter_id, chapter.motif),
                    now_ts=now_ts,
                )
                strengthened_edges.append(edge.edge_id)

        self.sleep_cycles += 1
        self.last_reweave_delta = round(coherence_shift + tension_shift, 4)
        return ReweaveMutation(
            chapter_ids=tuple(created_chapters),
            updated_fragment_ids=tuple(sorted(updated_fragments)),
            strengthened_edge_ids=tuple(strengthened_edges),
            coherence_shift=round(coherence_shift, 4),
            tension_shift=round(tension_shift, 4),
            self_drift={key: round(value, 4) for key, value in self_drift.items()},
            world_drift={key: round(value, 4) for key, value in world_drift.items()},
            notes=tuple(notes),
        )

    def _select_reweave_candidates(self, top_k: int = 24) -> tuple[Fragment, ...]:
        ranked = sorted(
            self.fragments.values(),
            key=lambda item: (
                0.30 * item.salience
                + 0.25 * item.unresolvedness
                + 0.20 * min(item.activation_count / 5.0, 1.0)
                + 0.15 * item.vividness
                + 0.10 * self._structural_pressure(item)
            ),
            reverse=True,
        )
        return tuple(ranked[:top_k])

    def _simple_cluster(self, candidates: Iterable[Fragment]) -> list[list[Fragment]]:
        remaining = list(candidates)
        groups: list[list[Fragment]] = []
        while remaining:
            seed = remaining.pop(0)
            cluster = [seed]
            keep: list[Fragment] = []
            for fragment in remaining:
                if self._affinity(seed, fragment) >= 0.45:
                    cluster.append(fragment)
                else:
                    keep.append(fragment)
            remaining = keep
            groups.append(cluster)
        return groups

    def _build_chapter(self, relation_id: str, cluster: list[Fragment], now_ts: float) -> Chapter:
        motif = self._infer_motif(cluster)
        title = self._chapter_title(cluster, motif)
        tension = sum(fragment.unresolvedness for fragment in cluster) / len(cluster)
        coherence = min(
            1.0,
            0.35 + self._cluster_keyword_overlap(cluster) + self._cluster_trace_overlap(cluster),
        )
        roles = self._assign_roles(cluster)
        return Chapter(
            chapter_id=f"chap_{uuid4().hex[:12]}",
            relation_id=relation_id,
            title=title,
            motif=motif,
            fragment_ids=tuple(fragment.fragment_id for fragment in cluster),
            roles=roles,
            tension=round(tension, 4),
            coherence=round(coherence, 4),
            created_at=now_ts,
            last_rewoven_at=now_ts,
        )

    def _assign_roles(self, cluster: list[Fragment]) -> dict[str, ChapterRole]:
        ordered = sorted(cluster, key=lambda item: item.created_at)
        roles: dict[str, ChapterRole] = {}
        if ordered:
            roles[ordered[0].fragment_id] = ChapterRole.SEED
            roles[ordered[-1].fragment_id] = ChapterRole.ANCHOR
        roles[max(cluster, key=lambda item: item.unresolvedness).fragment_id] = (
            ChapterRole.UNRESOLVED_KNOT
        )
        roles[max(cluster, key=lambda item: item.salience).fragment_id] = ChapterRole.TURNING_POINT
        return roles

    def _infer_motif(self, cluster: list[Fragment]) -> str:
        counter: dict[str, float] = {}
        for fragment in cluster:
            for trace in self.traces_for_fragment(fragment.fragment_id):
                counter[trace.channel.value] = (
                    counter.get(trace.channel.value, 0.0) + trace.intensity
                )
        top_channels = [
            name for name, _ in sorted(counter.items(), key=lambda item: item[1], reverse=True)[:2]
        ]
        if not top_channels:
            return "continuity"
        if "boundary" in top_channels and "hurt" in top_channels:
            return "boundary after hurt"
        if "repair" in top_channels or ("warmth" in top_channels and "distance" in top_channels):
            return "repair attempt"
        if "recognition" in top_channels:
            return "recognition motif"
        return " / ".join(top_channels)

    def _chapter_title(self, cluster: list[Fragment], motif: str) -> str:
        tokens = Counter(token for fragment in cluster for token in fragment.tags)
        anchor = tokens.most_common(2)
        if anchor:
            return f"{'/'.join(token for token, _ in anchor)} - {motif}"
        return f"chapter - {motif}"

    def _novelty_bonus(self, fragment: Fragment) -> float:
        age_hours = max(0.0, fragment.last_touched_at - fragment.created_at) / 3600.0
        return 0.2 if age_hours < 6.0 else 0.0

    def _structural_pressure(self, fragment: Fragment) -> float:
        return min(1.0, len(self.fragment_edges.get(fragment.fragment_id, ())) / 5.0)

    def _cluster_keyword_overlap(self, cluster: list[Fragment]) -> float:
        if len(cluster) < 2:
            return 0.0
        overlaps = [
            self._keyword_overlap(left.tags, right.tags) for left, right in combinations(cluster, 2)
        ]
        return sum(overlaps) / len(overlaps)

    def _cluster_trace_overlap(self, cluster: list[Fragment]) -> float:
        if len(cluster) < 2:
            return 0.0
        overlaps: list[float] = []
        for left, right in combinations(cluster, 2):
            left_channels = {trace.channel for trace in self.traces_for_fragment(left.fragment_id)}
            right_channels = {
                trace.channel for trace in self.traces_for_fragment(right.fragment_id)
            }
            overlaps.append(
                len(left_channels & right_channels) / max(1, len(left_channels | right_channels))
            )
        return sum(overlaps) / len(overlaps)

    def _affinity(self, left: Fragment, right: Fragment) -> float:
        temporal_distance = abs(left.created_at - right.created_at) / 3600.0
        temporal = math.exp(-temporal_distance / 18.0)
        return (
            0.30 * float(left.relation_id == right.relation_id)
            + 0.25 * self._keyword_overlap(left.tags, right.tags)
            + 0.25 * self._trace_overlap(left.fragment_id, right.fragment_id)
            + 0.20 * temporal
        )

    def _trace_overlap(self, left_fragment_id: str, right_fragment_id: str) -> float:
        left = {trace.channel for trace in self.traces_for_fragment(left_fragment_id)}
        right = {trace.channel for trace in self.traces_for_fragment(right_fragment_id)}
        return len(left & right) / max(1, len(left | right)) if (left or right) else 0.0

    def _keyword_overlap(self, left_tags: Iterable[str], right_tags: Iterable[str]) -> float:
        left = set(left_tags)
        right = set(right_tags)
        return len(left & right) / max(1, len(left | right)) if (left or right) else 0.0


def extract_tags(text: str) -> tuple[str, ...]:
    latin = re.findall(r"[A-Za-z]{3,}", text.lower())
    cjk = re.findall(r"[\u4e00-\u9fff]{1,4}", text)
    merged = [token.strip() for token in latin + cjk if token.strip()]
    if not merged:
        merged = [text[:10].strip() or "moment"]
    return tuple(sorted(set(merged[:12])))
