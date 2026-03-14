from __future__ import annotations

from dataclasses import dataclass

from .events import TouchSignal
from .memory import Chapter, Fragment, MemoryGraph
from .schema import TraceChannel


@dataclass(frozen=True, slots=True)
class ReweaveCluster:
    relation_id: str
    fragment_ids: tuple[str, ...]
    dominant_channels: tuple[TraceChannel, ...]
    score: float


@dataclass(frozen=True, slots=True)
class ReweaveResult:
    relation_id: str
    chapter_ids: tuple[str, ...]
    strengthened_edge_ids: tuple[str, ...]
    coherence_shift: float
    tension_shift: float
    relation_bias: float


class NarrativeReweaver:
    def select_candidates(self, memory: MemoryGraph, relation_id: str | None = None, limit: int = 16) -> list[Fragment]:
        return memory.select_reweave_candidates(relation_id=relation_id, limit=limit)

    def cluster(self, fragments: list[Fragment]) -> list[ReweaveCluster]:
        clusters: list[list[Fragment]] = []
        for fragment in fragments:
            placed = False
            for cluster in clusters:
                anchor = cluster[0]
                overlap = anchor.touch_signature.overlap(fragment.touch_signature)
                same_relation = anchor.relation_id == fragment.relation_id
                close_in_time = abs(anchor.created_at - fragment.created_at) <= 6000.0
                if same_relation and (overlap >= 0.25 or close_in_time):
                    cluster.append(fragment)
                    placed = True
                    break
            if not placed:
                clusters.append([fragment])

        wrapped: list[ReweaveCluster] = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            relation_id = cluster[0].relation_id
            channel_totals: dict[TraceChannel, float] = {}
            for fragment in cluster:
                for channel, value in fragment.touch_signature.weights.items():
                    channel_totals[channel] = channel_totals.get(channel, 0.0) + value
            dominant = tuple(
                channel
                for channel, _ in sorted(channel_totals.items(), key=lambda item: item[1], reverse=True)[:2]
            )
            score = sum(fragment.salience + fragment.unresolvedness for fragment in cluster) / len(cluster)
            wrapped.append(
                ReweaveCluster(
                    relation_id=relation_id,
                    fragment_ids=tuple(fragment.fragment_id for fragment in sorted(cluster, key=lambda item: item.created_at)),
                    dominant_channels=dominant,
                    score=score,
                )
            )
        wrapped.sort(key=lambda item: item.score, reverse=True)
        return wrapped

    def reweave_relation(self, memory: MemoryGraph, relation_id: str, now_ts: float) -> ReweaveResult:
        candidates = self.select_candidates(memory, relation_id=relation_id, limit=16)
        clusters = self.cluster(candidates)
        if not clusters:
            return ReweaveResult(
                relation_id=relation_id,
                chapter_ids=(),
                strengthened_edge_ids=(),
                coherence_shift=0.0,
                tension_shift=-0.05,
                relation_bias=0.0,
            )

        chapter_ids: list[str] = []
        edge_ids: list[str] = []
        coherence_shift = 0.0
        tension_shift = 0.0
        relation_bias = 0.0

        for cluster in clusters[:3]:
            synopsis = self._build_synopsis(memory, cluster)
            chapter = memory.create_chapter(
                relation_id=relation_id,
                fragment_ids=cluster.fragment_ids,
                dominant_channels=cluster.dominant_channels,
                tension=min(1.0, 0.40 + cluster.score * 0.35),
                synopsis=synopsis,
                now_ts=now_ts,
            )
            chapter_ids.append(chapter.chapter_id)
            edge_ids.extend(memory.strengthen_cluster_edges(cluster.fragment_ids, now_ts=now_ts))
            coherence_shift += 0.08 + 0.04 * len(cluster.fragment_ids)
            tension_shift += 0.10 if "hurt" in cluster.dominant_channels or "boundary" in cluster.dominant_channels else -0.05
            relation_bias += 0.08 if "warmth" in cluster.dominant_channels or "recognition" in cluster.dominant_channels else 0.0

        return ReweaveResult(
            relation_id=relation_id,
            chapter_ids=tuple(chapter_ids),
            strengthened_edge_ids=tuple(edge_ids),
            coherence_shift=min(coherence_shift, 0.45),
            tension_shift=max(min(tension_shift, 0.40), -0.30),
            relation_bias=min(relation_bias, 0.30),
        )

    def reweave_all(self, memory: MemoryGraph, now_ts: float) -> list[ReweaveResult]:
        relation_ids = sorted({fragment.relation_id for fragment in memory.fragments.values()})
        return [self.reweave_relation(memory, relation_id, now_ts=now_ts) for relation_id in relation_ids]

    def _build_synopsis(self, memory: MemoryGraph, cluster: ReweaveCluster) -> str:
        fragments = [memory.fragments[fragment_id] for fragment_id in cluster.fragment_ids]
        first = fragments[0].surface
        last = fragments[-1].surface
        channels = ", ".join(cluster.dominant_channels) if cluster.dominant_channels else "mixed"
        return f"thread from '{first}' toward '{last}', dominated by {channels}"
