from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations
from uuid import uuid4

from aurora.runtime.models import (
    Association,
    AssocKind,
    Fragment,
    Knot,
    Thread,
    Trace,
    TraceChannel,
)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(slots=True)
class MemoryStore:
    fragments: dict[str, Fragment] = field(default_factory=dict)
    traces: dict[str, Trace] = field(default_factory=dict)
    associations: dict[str, Association] = field(default_factory=dict)
    threads: dict[str, Thread] = field(default_factory=dict)
    knots: dict[str, Knot] = field(default_factory=dict)
    sleep_cycles: int = 0
    last_reweave_delta: float = 0.0

    def add_fragment(self, fragment: Fragment) -> None:
        self.fragments[fragment.fragment_id] = fragment

    def add_traces(self, traces: tuple[Trace, ...]) -> None:
        for trace in traces:
            self.traces[trace.trace_id] = trace

    def add_association(self, association: Association) -> None:
        self.associations[association.edge_id] = association

    def relation_fragments(self, relation_id: str) -> list[Fragment]:
        return [
            fragment for fragment in self.fragments.values() if fragment.relation_id == relation_id
        ]

    def relation_traces(self, relation_id: str) -> list[Trace]:
        return [trace for trace in self.traces.values() if trace.relation_id == relation_id]

    def strongest_channels(self, relation_id: str, limit: int = 3) -> tuple[TraceChannel, ...]:
        scores: dict[TraceChannel, float] = {}
        for trace in self.relation_traces(relation_id):
            scores.setdefault(trace.channel, 0.0)
            scores[trace.channel] += trace.intensity * trace.persistence
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return tuple(channel for channel, _ in ranked[:limit])

    def recent_fragments(self, relation_id: str, limit: int = 8) -> list[Fragment]:
        fragments = sorted(
            self.relation_fragments(relation_id),
            key=lambda fragment: fragment.last_touched_at,
            reverse=True,
        )
        return fragments[:limit]

    def recall(self, relation_id: str, limit: int = 4) -> tuple[Fragment, ...]:
        candidates = self.relation_fragments(relation_id)
        if not candidates:
            return ()

        channel_bias = set(self.strongest_channels(relation_id, limit=3))
        max_timestamp = max(fragment.last_touched_at for fragment in candidates)
        scored: list[tuple[float, Fragment]] = []
        for fragment in candidates:
            recency = 1.0 / (1.0 + max(0.0, max_timestamp - fragment.last_touched_at) / 600.0)
            channel_score = (
                1.0 if any(channel in channel_bias for channel in fragment.touch_channels) else 0.30
            )
            thread_bonus = 0.20 * sum(
                1 for thread in self.threads.values() if fragment.fragment_id in thread.fragment_ids
            )
            knot_penalty = 0.18 * sum(
                1 for knot in self.knots.values() if fragment.fragment_id in knot.fragment_ids
            )
            score = (
                0.28 * fragment.salience
                + 0.20 * fragment.activation
                + 0.18 * fragment.vividness
                + 0.14 * fragment.unresolvedness
                + 0.12 * recency
                + 0.08 * channel_score
                + thread_bonus
                - knot_penalty
            )
            scored.append((score, fragment))

        scored.sort(key=lambda item: item[0], reverse=True)
        return tuple(fragment for _, fragment in scored[:limit])

    def doze_consolidate(self, relation_id: str, now_ts: float) -> tuple[str, ...]:
        touched_ids: list[str] = []
        for fragment in self.recent_fragments(relation_id, limit=12):
            updated = replace(
                fragment,
                vividness=_clamp(fragment.vividness * 0.96, 0.0, 1.0),
                activation=_clamp(fragment.activation * 0.94 + 0.02, 0.0, 1.0),
                last_touched_at=now_ts,
            )
            self.fragments[fragment.fragment_id] = updated
            touched_ids.append(fragment.fragment_id)
        return tuple(touched_ids)

    def reweave(
        self,
        relation_id: str,
        now_ts: float,
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...], float]:
        candidates = sorted(
            self.relation_fragments(relation_id),
            key=lambda fragment: fragment.unresolvedness + fragment.activation + fragment.salience,
            reverse=True,
        )[:8]

        if not candidates:
            self.sleep_cycles += 1
            self.last_reweave_delta = 0.0
            return (), (), (), (), 0.0

        softened_ids: list[str] = []
        for fragment in candidates:
            updated = replace(
                fragment,
                vividness=_clamp(fragment.vividness * 0.82, 0.0, 1.0),
                salience=_clamp(fragment.salience * 0.86, 0.0, 1.0),
                unresolvedness=_clamp(fragment.unresolvedness * 0.93, 0.0, 1.0),
                activation=_clamp(fragment.activation * 0.88, 0.0, 1.0),
                last_touched_at=now_ts,
            )
            self.fragments[fragment.fragment_id] = updated
            softened_ids.append(fragment.fragment_id)

        edge_ids: list[str] = []
        for left, right in combinations(candidates[:5], 2):
            edge = Association(
                edge_id=f"edge_{uuid4().hex[:10]}",
                src_fragment_id=left.fragment_id,
                dst_fragment_id=right.fragment_id,
                kind=AssocKind.RESONANCE,
                weight=_clamp((left.activation + right.activation) / 2.0, 0.0, 1.0),
                evidence_count=1,
                created_at=now_ts,
                last_touched_at=now_ts,
            )
            self.associations[edge.edge_id] = edge
            edge_ids.append(edge.edge_id)

        channels = self.strongest_channels(relation_id, limit=2)
        thread = Thread(
            thread_id=f"thr_{uuid4().hex[:10]}",
            relation_id=relation_id,
            fragment_ids=tuple(fragment.fragment_id for fragment in candidates[:5]),
            motif_channels=channels,
            coherence=_clamp(
                sum(fragment.salience for fragment in candidates[:5]) / max(len(candidates[:5]), 1),
                0.0,
                1.0,
            ),
            tension=_clamp(
                sum(fragment.unresolvedness for fragment in candidates[:5])
                / max(len(candidates[:5]), 1),
                0.0,
                1.0,
            ),
            synopsis="sleep-woven thread",
            created_at=now_ts,
            last_rewoven_at=now_ts,
        )
        self.threads[thread.thread_id] = thread

        knot_ids: list[str] = []
        high_unresolved = [fragment for fragment in candidates if fragment.unresolvedness >= 0.6]
        if high_unresolved:
            knot = Knot(
                knot_id=f"knot_{uuid4().hex[:10]}",
                relation_id=relation_id,
                fragment_ids=tuple(fragment.fragment_id for fragment in high_unresolved[:4]),
                channel=channels[0] if channels else TraceChannel.AMBIENT,
                density=_clamp(
                    sum(fragment.activation for fragment in high_unresolved[:4])
                    / max(len(high_unresolved[:4]), 1),
                    0.0,
                    1.0,
                ),
                heat=_clamp(
                    sum(fragment.unresolvedness for fragment in high_unresolved[:4])
                    / max(len(high_unresolved[:4]), 1),
                    0.0,
                    1.0,
                ),
                created_at=now_ts,
                last_touched_at=now_ts,
            )
            self.knots[knot.knot_id] = knot
            knot_ids.append(knot.knot_id)

        self.sleep_cycles += 1
        delta = float(len(softened_ids) + len(edge_ids) + len(knot_ids)) / 24.0
        self.last_reweave_delta = delta
        return tuple(softened_ids), tuple(edge_ids), (thread.thread_id,), tuple(knot_ids), delta
