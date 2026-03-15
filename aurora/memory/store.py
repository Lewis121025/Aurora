from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Iterable
from uuid import uuid4

from aurora.memory.association import Association
from aurora.memory.fragment import Fragment
from aurora.memory.knot import Knot
from aurora.memory.tags import extract_tags
from aurora.memory.thread import Thread
from aurora.memory.trace import Trace
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

    def strengthen_association(
        self,
        src_fragment_id: str,
        dst_fragment_id: str,
        kind: AssocKind,
        weight: float,
        evidence: Iterable[str],
        now_ts: float,
    ) -> Association:
        existing_id = self._existing_edge_id(src_fragment_id, dst_fragment_id, kind=kind)
        if existing_id is None:
            return self.link_fragments(
                src_fragment_id=src_fragment_id,
                dst_fragment_id=dst_fragment_id,
                kind=kind,
                weight=weight,
                evidence=evidence,
                now_ts=now_ts,
            )
        edge = self.associations[existing_id]
        merged_evidence = tuple(dict.fromkeys([*edge.evidence, *tuple(evidence)]))
        updated = replace(
            edge,
            weight=clamp(max(edge.weight, weight)),
            evidence=merged_evidence,
            last_touched_at=now_ts,
        )
        self.associations[existing_id] = updated
        return updated

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

    def _existing_edge_id(
        self,
        left_fragment_id: str,
        right_fragment_id: str,
        kind: AssocKind | None = None,
    ) -> str | None:
        shared_edge_ids = self.fragment_edges.get(
            left_fragment_id, set()
        ) & self.fragment_edges.get(right_fragment_id, set())
        for edge_id in shared_edge_ids:
            edge = self.associations[edge_id]
            if kind is not None and edge.kind is not kind:
                continue
            return edge_id
        return None
