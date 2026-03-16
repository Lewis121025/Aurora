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

        self.relation_threads: dict[str, list[str]] = defaultdict(list)
        self.relation_knots: dict[str, list[str]] = defaultdict(list)

        self.sleep_cycles = 0
        self.last_sleep_at = 0.0

        self._dirty_fragments: set[str] = set()
        self._dirty_traces: set[str] = set()
        self._dirty_associations: set[str] = set()
        self._dirty_threads: set[str] = set()
        self._dirty_knots: set[str] = set()

        self._deleted_fragments: set[str] = set()
        self._deleted_traces: set[str] = set()
        self._deleted_associations: set[str] = set()
        self._deleted_threads: set[str] = set()
        self._deleted_knots: set[str] = set()

    def add_fragment(self, fragment: Fragment) -> None:
        self.fragments[fragment.fragment_id] = fragment
        self._dirty_fragments.add(fragment.fragment_id)
        if fragment.fragment_id not in self.relation_fragments[fragment.relation_id]:
            self.relation_fragments[fragment.relation_id].append(fragment.fragment_id)

    def add_trace(self, trace: Trace) -> None:
        self.traces[trace.trace_id] = trace
        self._dirty_traces.add(trace.trace_id)
        if trace.trace_id not in self.fragment_traces[trace.fragment_id]:
            self.fragment_traces[trace.fragment_id].append(trace.trace_id)

    def add_association(self, edge: Association) -> None:
        self.associations[edge.edge_id] = edge
        self._dirty_associations.add(edge.edge_id)
        self.fragment_edges[edge.src_fragment_id].add(edge.edge_id)
        self.fragment_edges[edge.dst_fragment_id].add(edge.edge_id)

    def add_thread(self, thread: Thread) -> None:
        self.threads[thread.thread_id] = thread
        self._dirty_threads.add(thread.thread_id)
        if thread.thread_id not in self.relation_threads[thread.relation_id]:
            self.relation_threads[thread.relation_id].append(thread.thread_id)
        for fragment_id in thread.fragment_ids:
            fragment = self.fragments[fragment_id]
            if thread.thread_id not in fragment.thread_ids:
                self.fragments[fragment_id] = replace(
                    fragment,
                    thread_ids=tuple(sorted(fragment.thread_ids + (thread.thread_id,))),
                )
                self._dirty_fragments.add(fragment_id)

    def add_knot(self, knot: Knot) -> None:
        self.knots[knot.knot_id] = knot
        self._dirty_knots.add(knot.knot_id)
        if knot.knot_id not in self.relation_knots[knot.relation_id]:
            self.relation_knots[knot.relation_id].append(knot.knot_id)
        for fragment_id in knot.fragment_ids:
            fragment = self.fragments[fragment_id]
            if knot.knot_id not in fragment.knot_ids:
                self.fragments[fragment_id] = replace(
                    fragment,
                    knot_ids=tuple(sorted(fragment.knot_ids + (knot.knot_id,))),
                )
                self._dirty_fragments.add(fragment_id)

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
        self._dirty_associations.add(existing_id)
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
        return tuple(
            self.threads[tid] for tid in self.relation_threads.get(relation_id, ())
            if tid in self.threads
        )

    def knots_for_relation(self, relation_id: str) -> tuple[Knot, ...]:
        return tuple(
            self.knots[kid] for kid in self.relation_knots.get(relation_id, ())
            if kid in self.knots
        )

    def find_matching_thread(
        self, relation_id: str, fragment_ids: tuple[str, ...], overlap: float = 0.75,
    ) -> Thread | None:
        target = set(fragment_ids)
        for thread in self.threads_for_relation(relation_id):
            existing = set(thread.fragment_ids)
            if len(target & existing) / max(1, len(target | existing)) >= overlap:
                return thread
        return None

    def find_matching_knot(
        self, relation_id: str, fragment_ids: tuple[str, ...], overlap: float = 0.75,
    ) -> Knot | None:
        target = set(fragment_ids)
        for knot in self.knots_for_relation(relation_id):
            existing = set(knot.fragment_ids)
            if len(target & existing) / max(1, len(target | existing)) >= overlap:
                return knot
        return None

    def update_thread(self, thread: Thread) -> None:
        old = self.threads.get(thread.thread_id)
        self.threads[thread.thread_id] = thread
        self._dirty_threads.add(thread.thread_id)
        if old is None:
            return
        new_fids = set(thread.fragment_ids) - set(old.fragment_ids)
        for fid in new_fids:
            fragment = self.fragments[fid]
            if thread.thread_id not in fragment.thread_ids:
                self.fragments[fid] = replace(
                    fragment,
                    thread_ids=tuple(sorted(fragment.thread_ids + (thread.thread_id,))),
                )
                self._dirty_fragments.add(fid)

    def update_knot(self, knot: Knot) -> None:
        old = self.knots.get(knot.knot_id)
        self.knots[knot.knot_id] = knot
        self._dirty_knots.add(knot.knot_id)
        if old is None:
            return
        new_fids = set(knot.fragment_ids) - set(old.fragment_ids)
        for fid in new_fids:
            fragment = self.fragments[fid]
            if knot.knot_id not in fragment.knot_ids:
                self.fragments[fid] = replace(
                    fragment,
                    knot_ids=tuple(sorted(fragment.knot_ids + (knot.knot_id,))),
                )
                self._dirty_fragments.add(fid)

    def touch_fragment(self, fragment_id: str, at: float, delta_salience: float = 0.08) -> None:
        self.fragments[fragment_id] = self.fragments[fragment_id].touched(
            at=at,
            delta_salience=delta_salience,
        )
        self._dirty_fragments.add(fragment_id)

    def set_fragment(self, fragment_id: str, fragment: Fragment) -> None:
        self.fragments[fragment_id] = fragment
        self._dirty_fragments.add(fragment_id)

    def set_trace(self, trace_id: str, trace: Trace) -> None:
        self.traces[trace_id] = trace
        self._dirty_traces.add(trace_id)

    def remove_fragment(self, fragment_id: str) -> None:
        fragment = self.fragments.pop(fragment_id, None)
        if fragment is None:
            return
        self._deleted_fragments.add(fragment_id)
        self._dirty_fragments.discard(fragment_id)
        rel_list = self.relation_fragments.get(fragment.relation_id)
        if rel_list and fragment_id in rel_list:
            rel_list.remove(fragment_id)
        self.fragment_traces.pop(fragment_id, None)
        self.fragment_edges.pop(fragment_id, None)

    def remove_trace(self, trace_id: str) -> None:
        trace = self.traces.pop(trace_id, None)
        if trace is None:
            return
        self._deleted_traces.add(trace_id)
        self._dirty_traces.discard(trace_id)
        ft = self.fragment_traces.get(trace.fragment_id)
        if ft and trace_id in ft:
            ft.remove(trace_id)

    def remove_association(self, edge_id: str) -> None:
        edge = self.associations.pop(edge_id, None)
        if edge is None:
            return
        self._deleted_associations.add(edge_id)
        self._dirty_associations.discard(edge_id)
        src_edges = self.fragment_edges.get(edge.src_fragment_id)
        if src_edges:
            src_edges.discard(edge_id)
        dst_edges = self.fragment_edges.get(edge.dst_fragment_id)
        if dst_edges:
            dst_edges.discard(edge_id)

    def remove_thread(self, thread_id: str) -> None:
        thread = self.threads.pop(thread_id, None)
        if thread is None:
            return
        self._deleted_threads.add(thread_id)
        self._dirty_threads.discard(thread_id)
        rt = self.relation_threads.get(thread.relation_id)
        if rt and thread_id in rt:
            rt.remove(thread_id)

    def remove_knot(self, knot_id: str) -> None:
        knot = self.knots.pop(knot_id, None)
        if knot is None:
            return
        self._deleted_knots.add(knot_id)
        self._dirty_knots.discard(knot_id)
        rk = self.relation_knots.get(knot.relation_id)
        if rk and knot_id in rk:
            rk.remove(knot_id)

    def clear_dirty(self) -> None:
        self._dirty_fragments.clear()
        self._dirty_traces.clear()
        self._dirty_associations.clear()
        self._dirty_threads.clear()
        self._dirty_knots.clear()
        self._deleted_fragments.clear()
        self._deleted_traces.clear()
        self._deleted_associations.clear()
        self._deleted_threads.clear()
        self._deleted_knots.clear()

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
