from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from itertools import combinations
import math
import re
from typing import Iterable
from uuid import uuid4

from aurora.memory.association import Association
from aurora.memory.fragment import Fragment
from aurora.memory.knot import Knot
from aurora.memory.reweave import SleepMutation
from aurora.memory.thread import Thread
from aurora.memory.trace import Trace
from aurora.relation.formation import RelationFormation
from aurora.runtime.contracts import AssocKind, TraceChannel, clamp


class MemoryStore:
    def __init__(self) -> None:
        self.fragments: dict[str, Fragment] = {}
        self.traces: dict[str, Trace] = {}
        self.associations: dict[str, Association] = {}
        self.threads: dict[str, Thread] = {}
        self.knots: dict[str, Knot] = {}

        self.fragment_traces: dict[str, list[str]] = defaultdict(list)
        self.relation_fragments: dict[str, list[str]] = defaultdict(list)
        self.fragment_edges: dict[str, set[str]] = defaultdict(set)

        self.sleep_cycles = 0
        self.last_sleep_at = 0.0

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

    def add_thread(self, thread: Thread) -> None:
        self.threads[thread.thread_id] = thread
        for fragment_id in thread.fragment_ids:
            fragment = self.fragments[fragment_id]
            if thread.thread_id not in fragment.thread_ids:
                self.fragments[fragment_id] = replace(
                    fragment,
                    thread_ids=tuple(sorted(fragment.thread_ids + (thread.thread_id,))),
                )

    def add_knot(self, knot: Knot) -> None:
        self.knots[knot.knot_id] = knot
        for fragment_id in knot.fragment_ids:
            fragment = self.fragments[fragment_id]
            if knot.knot_id not in fragment.knot_ids:
                self.fragments[fragment_id] = replace(
                    fragment,
                    knot_ids=tuple(sorted(fragment.knot_ids + (knot.knot_id,))),
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
            thread_ids=(),
            knot_ids=(),
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
        carry: float | None = None,
    ) -> Trace:
        final_intensity = clamp(intensity)
        trace = Trace(
            trace_id=f"trace_{uuid4().hex[:12]}",
            relation_id=relation_id,
            fragment_id=fragment_id,
            channel=channel,
            intensity=final_intensity,
            carry=clamp(carry if carry is not None else 0.35 + final_intensity * 0.5),
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
            self.traces[trace_id] for trace_id in self.fragment_traces.get(fragment_id, ())
        )

    def fragments_for_relation(self, relation_id: str) -> tuple[Fragment, ...]:
        return tuple(
            self.fragments[fragment_id]
            for fragment_id in self.relation_fragments.get(relation_id, ())
            if fragment_id in self.fragments
        )

    def threads_for_relation(self, relation_id: str) -> tuple[Thread, ...]:
        return tuple(item for item in self.threads.values() if item.relation_id == relation_id)

    def knots_for_relation(self, relation_id: str) -> tuple[Knot, ...]:
        return tuple(item for item in self.knots.values() if item.relation_id == relation_id)

    def touch_fragment(self, fragment_id: str, at: float, delta_salience: float = 0.08) -> None:
        self.fragments[fragment_id] = self.fragments[fragment_id].touched(
            at=at,
            delta_salience=delta_salience,
        )

    def recent_recall(self, relation_id: str, limit: int = 8) -> tuple[Fragment, ...]:
        ranked = sorted(
            self.fragments_for_relation(relation_id),
            key=lambda item: (
                0.36 * item.salience
                + 0.26 * item.unresolvedness
                + 0.14 * min(item.activation_count / 4.0, 1.0)
                + 0.12 * self._structural_pressure(item)
                + 0.12 * min(1.0, len(item.thread_ids) * 0.4 + len(item.knot_ids) * 0.6)
            ),
            reverse=True,
        )
        return tuple(ranked[:limit])

    def build_activation_channels(
        self, fragments: tuple[Fragment, ...]
    ) -> tuple[TraceChannel, ...]:
        scores: dict[TraceChannel, float] = {}
        for fragment in fragments:
            for trace in self.traces_for_fragment(fragment.fragment_id):
                scores[trace.channel] = (
                    scores.get(trace.channel, 0.0) + trace.intensity * trace.carry
                )
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return tuple(channel for channel, _ in ranked[:4])

    def decay_for_doze(self, now_ts: float) -> None:
        for fragment_id, fragment in list(self.fragments.items()):
            hours = max(0.0, (now_ts - fragment.last_touched_at) / 3600.0)
            salience_drop = min(0.12, hours * 0.012)
            unresolved_drop = min(0.09, hours * 0.01)
            self.fragments[fragment_id] = replace(
                fragment,
                salience=clamp(fragment.salience - salience_drop),
                unresolvedness=clamp(fragment.unresolvedness - unresolved_drop),
                last_touched_at=now_ts,
            )
        for trace_id, trace in list(self.traces.items()):
            hours = max(0.0, (now_ts - trace.last_touched_at) / 3600.0)
            next_intensity = clamp(trace.intensity - trace.carry * min(0.2, hours / 48.0))
            self.traces[trace_id] = replace(trace, intensity=next_intensity, last_touched_at=now_ts)

    def reweave(
        self,
        relation_formations: dict[str, RelationFormation],
        now_ts: float,
        pending_relations: tuple[str, ...] | None = None,
    ) -> SleepMutation:
        relation_ids = pending_relations or tuple(sorted(self.relation_fragments.keys()))

        created_thread_ids: list[str] = []
        created_knot_ids: list[str] = []
        strengthened_edge_ids: list[str] = []
        softened_fragment_ids: set[str] = set()
        affected_relation_ids: list[str] = []
        recall_bias: dict[str, tuple[str, ...]] = {}

        for relation_id in relation_ids:
            candidates = self._select_reweave_candidates(relation_id=relation_id)
            clusters = [
                cluster for cluster in self._simple_cluster(candidates) if len(cluster) >= 2
            ]
            if not clusters:
                continue

            affected_relation_ids.append(relation_id)
            relation_thread_ids: list[str] = []
            relation_knot_ids: list[str] = []

            for cluster in clusters[:3]:
                dominant_channels = self._cluster_dominant_channels(cluster)
                tension = sum(item.unresolvedness for item in cluster) / len(cluster)
                coherence = min(
                    1.0,
                    0.38
                    + self._cluster_keyword_overlap(cluster)
                    + self._cluster_trace_overlap(cluster),
                )

                thread = self._build_thread(
                    relation_id=relation_id,
                    cluster=cluster,
                    dominant_channels=dominant_channels,
                    tension=tension,
                    coherence=coherence,
                    now_ts=now_ts,
                )
                self.add_thread(thread)
                created_thread_ids.append(thread.thread_id)
                relation_thread_ids.append(thread.thread_id)

                knot: Knot | None = None
                if tension >= 0.58:
                    knot = self._build_knot(
                        relation_id=relation_id,
                        cluster=cluster,
                        dominant_channels=dominant_channels,
                        intensity=tension,
                        now_ts=now_ts,
                    )
                    self.add_knot(knot)
                    created_knot_ids.append(knot.knot_id)
                    relation_knot_ids.append(knot.knot_id)

                for fragment in cluster:
                    softened_fragment_ids.add(fragment.fragment_id)
                    self.fragments[fragment.fragment_id] = self.fragments[
                        fragment.fragment_id
                    ].touched(
                        at=now_ts,
                        delta_salience=0.06,
                        delta_unresolved=-0.09,
                    )

                edge_kind = AssocKind.KNOT if knot is not None else AssocKind.THREAD
                evidence_token = knot.knot_id if knot is not None else thread.thread_id
                for left, right in combinations(cluster, 2):
                    edge = self.link_fragments(
                        src_fragment_id=left.fragment_id,
                        dst_fragment_id=right.fragment_id,
                        kind=edge_kind,
                        weight=0.52 + 0.35 * self._affinity(left, right),
                        evidence=(evidence_token,),
                        now_ts=now_ts,
                    )
                    strengthened_edge_ids.append(edge.edge_id)

            recall_bias[relation_id] = tuple(relation_thread_ids[-4:])
            formation = relation_formations.get(relation_id)
            if formation is not None:
                formation.absorb_sleep(
                    thread_ids=tuple(relation_thread_ids),
                    knot_ids=tuple(relation_knot_ids),
                    now_ts=now_ts,
                )

        if created_thread_ids or created_knot_ids:
            self.sleep_cycles += 1
            self.last_sleep_at = now_ts

        return SleepMutation(
            created_thread_ids=tuple(created_thread_ids),
            created_knot_ids=tuple(created_knot_ids),
            strengthened_edge_ids=tuple(strengthened_edge_ids),
            softened_fragment_ids=tuple(sorted(softened_fragment_ids)),
            affected_relation_ids=tuple(affected_relation_ids),
            recall_bias=recall_bias,
        )

    def _select_reweave_candidates(self, relation_id: str, top_k: int = 24) -> tuple[Fragment, ...]:
        ranked = sorted(
            self.fragments_for_relation(relation_id),
            key=lambda item: (
                0.36 * item.salience
                + 0.28 * item.unresolvedness
                + 0.22 * min(item.activation_count / 5.0, 1.0)
                + 0.14 * self._structural_pressure(item)
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

    def _build_thread(
        self,
        relation_id: str,
        cluster: list[Fragment],
        dominant_channels: tuple[TraceChannel, ...],
        tension: float,
        coherence: float,
        now_ts: float,
    ) -> Thread:
        return Thread(
            thread_id=f"thread_{uuid4().hex[:12]}",
            relation_id=relation_id,
            fragment_ids=tuple(fragment.fragment_id for fragment in cluster),
            dominant_channels=dominant_channels,
            tension=round(clamp(tension), 4),
            coherence=round(clamp(coherence), 4),
            created_at=now_ts,
            last_rewoven_at=now_ts,
        )

    def _build_knot(
        self,
        relation_id: str,
        cluster: list[Fragment],
        dominant_channels: tuple[TraceChannel, ...],
        intensity: float,
        now_ts: float,
    ) -> Knot:
        return Knot(
            knot_id=f"knot_{uuid4().hex[:12]}",
            relation_id=relation_id,
            fragment_ids=tuple(fragment.fragment_id for fragment in cluster),
            dominant_channels=dominant_channels,
            intensity=round(clamp(intensity), 4),
            resolved=False,
            created_at=now_ts,
            last_rewoven_at=now_ts,
        )

    def _structural_pressure(self, fragment: Fragment) -> float:
        return min(1.0, len(self.fragment_edges.get(fragment.fragment_id, ())) / 6.0)

    def _cluster_dominant_channels(self, cluster: list[Fragment]) -> tuple[TraceChannel, ...]:
        totals: dict[TraceChannel, float] = {}
        for fragment in cluster:
            for trace in self.traces_for_fragment(fragment.fragment_id):
                totals[trace.channel] = totals.get(trace.channel, 0.0) + trace.intensity
        ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
        return tuple(channel for channel, _ in ranked[:2])

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
        temporal = math.exp(-temporal_distance / 16.0)
        return (
            0.30 * float(left.relation_id == right.relation_id)
            + 0.24 * self._keyword_overlap(left.tags, right.tags)
            + 0.26 * self._trace_overlap(left.fragment_id, right.fragment_id)
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
        merged = [text[:12].strip() or "moment"]
    counter = Counter(merged)
    return tuple(sorted(token for token, _ in counter.most_common(12)))
