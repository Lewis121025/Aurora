from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations
from uuid import uuid4

from models_v2_example import AssociationEdge, Chapter, Fragment, Tone, TraceResidue


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _tokenize(text: str) -> set[str]:
    return {part.strip(".,!?;:()[]{}").lower() for part in text.split() if part.strip()}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / float(len(union))


@dataclass(frozen=True, slots=True)
class ReweaveResult:
    delta: float
    chapter_ids: tuple[str, ...]
    coherence_shift: float
    tension_shift: float
    relation_bias: float


@dataclass(slots=True)
class MemoryStoreV2:
    """
    一个可独立运行的示例 store。
    你可以把里面的方法迁移到你当前的 aurora.memory.store。
    """
    fragments: dict[str, Fragment] = field(default_factory=dict)
    traces: list[TraceResidue] = field(default_factory=list)
    associations: list[AssociationEdge] = field(default_factory=list)
    chapters: dict[str, Chapter] = field(default_factory=dict)
    fragment_reactivation_count: dict[str, int] = field(default_factory=dict)
    sleep_cycles: int = 0
    last_reweave_delta: float = 0.0

    def add_fragment(self, fragment: Fragment) -> None:
        self.fragments[fragment.fragment_id] = fragment
        self.fragment_reactivation_count.setdefault(fragment.fragment_id, 0)

    def add_trace(self, trace: TraceResidue) -> None:
        self.traces.append(trace)

    def add_association(self, edge: AssociationEdge) -> None:
        self.associations.append(edge)

    def reactivation_count(self, fragment_id: str) -> int:
        return self.fragment_reactivation_count.get(fragment_id, 0)

    def note_recall(self, fragment_id: str) -> None:
        self.fragment_reactivation_count[fragment_id] = self.reactivation_count(fragment_id) + 1

    def traces_for_fragment(self, fragment_id: str) -> list[TraceResidue]:
        return [trace for trace in self.traces if trace.fragment_id == fragment_id]

    def trace_strength(self, fragment_id: str) -> float:
        relevant = self.traces_for_fragment(fragment_id)
        if not relevant:
            return 0.0
        total = 0.0
        for trace in relevant:
            total += trace.intensity * trace.decay
        return _clamp(total / float(len(relevant)), 0.0, 1.0)

    def trace_channel_vector(self, fragment_id: str) -> dict[str, float]:
        scores: dict[str, float] = {}
        for trace in self.traces_for_fragment(fragment_id):
            scores.setdefault(trace.channel, 0.0)
            scores[trace.channel] += trace.intensity * trace.decay
        return scores

    def association_strength_for_fragment(self, fragment_id: str) -> float:
        relevant = [
            edge.weight
            for edge in self.associations
            if edge.source_fragment_id == fragment_id or edge.target_fragment_id == fragment_id
        ]
        if not relevant:
            return 0.0
        return _clamp(sum(relevant) / float(len(relevant)), 0.0, 1.0)

    def direct_association(self, left_id: str, right_id: str) -> float:
        best = 0.0
        for edge in self.associations:
            if {
                edge.source_fragment_id,
                edge.target_fragment_id,
            } == {left_id, right_id}:
                best = max(best, edge.weight)
        return best

    def select_candidates(self, relation_id: str | None, top_k: int = 24) -> list[Fragment]:
        scored: list[tuple[float, Fragment]] = []
        for fragment in self.fragments.values():
            relation_bonus = 0.15 if relation_id and fragment.relation_id == relation_id else 0.0
            score = (
                0.35 * fragment.salience
                + 0.25 * fragment.unresolvedness
                + 0.15 * self.trace_strength(fragment.fragment_id)
                + 0.10 * self.association_strength_for_fragment(fragment.fragment_id)
                + 0.15 * _clamp(self.reactivation_count(fragment.fragment_id) / 5.0, 0.0, 1.0)
                + relation_bonus
            )
            scored.append((score, fragment))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [fragment for _, fragment in scored[:top_k]]

    def cluster_candidates(self, candidates: list[Fragment], threshold: float = 0.50) -> list[list[Fragment]]:
        """
        一个很简单的 O(n^2) 聚团示例：
        够用来表达 Aurora 里的“局部叙事子图”概念。
        """
        groups: list[list[Fragment]] = []
        unassigned = list(candidates)

        while unassigned:
            seed = unassigned.pop(0)
            cluster = [seed]
            changed = True
            while changed:
                changed = False
                remaining: list[Fragment] = []
                for fragment in unassigned:
                    if any(self._pair_similarity(fragment, member) >= threshold for member in cluster):
                        cluster.append(fragment)
                        changed = True
                    else:
                        remaining.append(fragment)
                unassigned = remaining
            groups.append(cluster)

        # 过滤掉既不成簇、也不够 unresolved 的单点噪声
        return [
            cluster
            for cluster in groups
            if len(cluster) >= 2 or max(fragment.unresolvedness for fragment in cluster) >= 0.65
        ]

    def build_chapter_from_cluster(
        self,
        cluster: list[Fragment],
        relation_id: str | None,
        now_ts: float,
    ) -> Chapter:
        dominant_channel = self._dominant_channel(cluster)
        title_map = {
            "warmth": "a thread of warmth and return",
            "hurt": "a thread of hurt and caution",
            "recognition": "a thread of recognition",
            "distance": "a thread of distance",
            "curiosity": "a thread of wondering",
            "boundary": "a thread of boundary",
        }
        title = title_map.get(dominant_channel, "a narrative thread")
        coherence = _clamp(sum(fragment.salience for fragment in cluster) / float(len(cluster)), 0.0, 1.0)
        tension = _clamp(sum(fragment.unresolvedness for fragment in cluster) / float(len(cluster)), 0.0, 1.0)

        anchors = tuple(fragment.fragment_id for fragment in sorted(cluster, key=lambda item: item.salience, reverse=True)[:4])
        return Chapter(
            chapter_id=f"chap_{uuid4().hex[:10]}",
            relation_id=relation_id,
            title=title,
            motif=dominant_channel,
            anchor_fragment_ids=anchors,
            coherence=coherence,
            tension=tension,
            created_at=now_ts,
            updated_at=now_ts,
        )

    def apply_chapter(self, chapter: Chapter, cluster: list[Fragment], relation_strength: float, now_ts: float) -> float:
        total_change = 0.0
        self.chapters[chapter.chapter_id] = chapter

        for fragment in cluster:
            old = self.fragments[fragment.fragment_id]
            target_salience = _clamp(
                0.60 * old.salience + 0.25 * chapter.coherence + 0.15 * relation_strength,
                0.0,
                1.0,
            )
            # unresolvedness 不是简单下降；有些 tension 会被保留为未来召回入口
            target_unresolved = _clamp(
                0.65 * old.unresolvedness + 0.35 * chapter.tension,
                0.0,
                1.0,
            )
            new_fragment = replace(
                old,
                salience=target_salience,
                unresolvedness=target_unresolved,
                chapter_ids=tuple(sorted(set(old.chapter_ids) | {chapter.chapter_id})),
            )
            self.fragments[new_fragment.fragment_id] = new_fragment
            total_change += abs(new_fragment.salience - old.salience)
            total_change += abs(new_fragment.unresolvedness - old.unresolvedness)

        # 为簇内片段重写边：共鸣 / 对比 / 关系修复 / 边界
        for left, right in combinations(cluster, 2):
            kind = self._edge_kind(left.fragment_id, right.fragment_id)
            base = self.direct_association(left.fragment_id, right.fragment_id)
            new_weight = _clamp(
                0.55 * base
                + 0.25 * self._shared_trace_strength(left.fragment_id, right.fragment_id)
                + 0.20 * chapter.coherence,
                0.0,
                1.0,
            )
            self._upsert_edge(
                source_fragment_id=left.fragment_id,
                target_fragment_id=right.fragment_id,
                kind=kind,
                weight=new_weight,
                now_ts=now_ts,
            )
        return total_change

    def _pair_similarity(self, left: Fragment, right: Fragment) -> float:
        score = 0.0
        if left.relation_id and left.relation_id == right.relation_id:
            score += 0.20

        token_overlap = _jaccard(_tokenize(left.surface), _tokenize(right.surface))
        score += 0.25 * token_overlap
        score += 0.30 * self._shared_trace_strength(left.fragment_id, right.fragment_id)
        score += 0.20 * self.direct_association(left.fragment_id, right.fragment_id)

        chapter_overlap = len(set(left.chapter_ids) & set(right.chapter_ids))
        if chapter_overlap > 0:
            score += 0.05 * min(2, chapter_overlap)

        return _clamp(score, 0.0, 1.0)

    def _dominant_channel(self, cluster: list[Fragment]) -> str:
        channel_scores: dict[str, float] = {}
        for fragment in cluster:
            for channel, value in self.trace_channel_vector(fragment.fragment_id).items():
                channel_scores.setdefault(channel, 0.0)
                channel_scores[channel] += value
        if not channel_scores:
            return "recognition"
        return max(channel_scores.items(), key=lambda item: item[1])[0]

    def _shared_trace_strength(self, left_id: str, right_id: str) -> float:
        left = self.trace_channel_vector(left_id)
        right = self.trace_channel_vector(right_id)
        channels = set(left) | set(right)
        if not channels:
            return 0.0
        shared = 0.0
        for channel in channels:
            shared += min(left.get(channel, 0.0), right.get(channel, 0.0))
        return _clamp(shared / float(len(channels)), 0.0, 1.0)

    def _edge_kind(self, left_id: str, right_id: str) -> str:
        left = self.trace_channel_vector(left_id)
        right = self.trace_channel_vector(right_id)
        if ("hurt" in left and "warmth" in right) or ("warmth" in left and "hurt" in right):
            return "repair"
        if ("boundary" in left) or ("boundary" in right):
            return "boundary"
        if ("distance" in left and "warmth" in right) or ("warmth" in left and "distance" in right):
            return "contrast"
        return "resonance"

    def _upsert_edge(
        self,
        source_fragment_id: str,
        target_fragment_id: str,
        kind: str,
        weight: float,
        now_ts: float,
    ) -> None:
        for index, edge in enumerate(self.associations):
            if {edge.source_fragment_id, edge.target_fragment_id} == {
                source_fragment_id,
                target_fragment_id,
            }:
                self.associations[index] = replace(
                    edge,
                    kind=kind,
                    weight=weight,
                    last_touched_at=now_ts,
                )
                return
        self.associations.append(
            AssociationEdge(
                edge_id=f"edge_{uuid4().hex[:10]}",
                source_fragment_id=source_fragment_id,
                target_fragment_id=target_fragment_id,
                kind=kind,
                weight=weight,
                created_at=now_ts,
                last_touched_at=now_ts,
            )
        )


def run_sleep_reweave_v2(
    store: MemoryStoreV2,
    relation_id: str | None,
    relation_tone: Tone,
    relation_strength: float,
    now_ts: float,
) -> ReweaveResult:
    if not store.fragments:
        store.sleep_cycles += 1
        store.last_reweave_delta = 0.0
        return ReweaveResult(
            delta=0.0,
            chapter_ids=(),
            coherence_shift=0.0,
            tension_shift=0.0,
            relation_bias=0.0,
        )

    candidates = store.select_candidates(relation_id=relation_id, top_k=24)
    clusters = store.cluster_candidates(candidates)

    chapter_ids: list[str] = []
    total_change = 0.0
    coherence_total = 0.0
    tension_total = 0.0

    for cluster in clusters:
        chapter = store.build_chapter_from_cluster(cluster=cluster, relation_id=relation_id, now_ts=now_ts)
        chapter_ids.append(chapter.chapter_id)
        total_change += store.apply_chapter(
            chapter=chapter,
            cluster=cluster,
            relation_strength=relation_strength,
            now_ts=now_ts,
        )
        coherence_total += chapter.coherence
        tension_total += chapter.tension

    store.sleep_cycles += 1
    normalizer = max(len(candidates) * 2, 1)
    delta = total_change / float(normalizer)
    store.last_reweave_delta = delta

    relation_bias = {
        "warm": +0.25,
        "neutral": 0.0,
        "cold": -0.18,
        "boundary": -0.28,
    }[relation_tone] * relation_strength

    chapter_count = max(len(chapter_ids), 1)
    coherence_shift = _clamp(coherence_total / float(chapter_count), 0.0, 1.0)
    tension_shift = _clamp(tension_total / float(chapter_count), 0.0, 1.0)

    return ReweaveResult(
        delta=delta,
        chapter_ids=tuple(chapter_ids),
        coherence_shift=coherence_shift,
        tension_shift=tension_shift,
        relation_bias=relation_bias,
    )
