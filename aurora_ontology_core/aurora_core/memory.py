from __future__ import annotations

from dataclasses import dataclass, replace

from .events import TouchSignal, make_id
from .schema import AssociationKind, ChapterStatus, Speaker, TraceChannel


@dataclass(frozen=True, slots=True)
class Fragment:
    fragment_id: str
    relation_id: str
    turn_id: str | None
    speaker: Speaker
    surface: str
    created_at: float
    vividness: float
    salience: float
    unresolvedness: float
    touch_signature: TouchSignal
    chapter_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TraceResidue:
    trace_id: str
    relation_id: str
    fragment_id: str
    channel: TraceChannel
    intensity: float
    carry: float
    created_at: float
    last_touched_at: float


@dataclass(frozen=True, slots=True)
class AssociationEdge:
    edge_id: str
    src_fragment_id: str
    dst_fragment_id: str
    kind: AssociationKind
    weight: float
    evidence: str
    last_touched_at: float


@dataclass(frozen=True, slots=True)
class Chapter:
    chapter_id: str
    relation_id: str
    title: str
    fragment_ids: tuple[str, ...]
    dominant_channels: tuple[TraceChannel, ...]
    tension: float
    status: ChapterStatus
    synopsis: str
    created_at: float
    last_rewoven_at: float


class MemoryGraph:
    """Object ontology memory core.

    It stores fragments, trace residues, typed associations, and chapters.
    There is no profile object and no concatenated-context memory surface here.
    """

    def __init__(self) -> None:
        self.fragments: dict[str, Fragment] = {}
        self.traces: dict[str, TraceResidue] = {}
        self.associations: dict[str, AssociationEdge] = {}
        self.chapters: dict[str, Chapter] = {}
        self._relation_fragments: dict[str, list[str]] = {}

    def add_fragment(
        self,
        *,
        relation_id: str,
        speaker: Speaker,
        surface: str,
        created_at: float,
        turn_id: str | None,
        touch_signature: TouchSignal,
        tags: tuple[str, ...] = (),
    ) -> Fragment:
        vividness = min(1.0, 0.35 + 0.45 * touch_signature.total_intensity())
        salience = min(1.0, 0.25 + 0.35 * touch_signature.total_intensity())
        unresolvedness = min(1.0, 0.15 + 0.30 * touch_signature.weights.get("hurt", 0.0) + 0.30 * touch_signature.weights.get("boundary", 0.0))

        fragment = Fragment(
            fragment_id=make_id("frag"),
            relation_id=relation_id,
            turn_id=turn_id,
            speaker=speaker,
            surface=surface,
            created_at=created_at,
            vividness=vividness,
            salience=salience,
            unresolvedness=unresolvedness,
            touch_signature=touch_signature,
            chapter_ids=(),
            tags=tags,
        )
        self.fragments[fragment.fragment_id] = fragment
        self._relation_fragments.setdefault(relation_id, []).append(fragment.fragment_id)

        for channel, intensity in touch_signature.weights.items():
            if intensity <= 0.0:
                continue
            self.add_trace(
                relation_id=relation_id,
                fragment_id=fragment.fragment_id,
                channel=channel,
                intensity=intensity,
                created_at=created_at,
            )

        same_relation = self.fragments_for_relation(relation_id)
        if len(same_relation) >= 2:
            prev = same_relation[-2]
            self.connect(
                src_fragment_id=prev.fragment_id,
                dst_fragment_id=fragment.fragment_id,
                kind="temporal",
                weight=0.35,
                evidence="same_relation_sequence",
                now_ts=created_at,
            )
        return fragment

    def add_trace(
        self,
        *,
        relation_id: str,
        fragment_id: str,
        channel: TraceChannel,
        intensity: float,
        created_at: float,
    ) -> TraceResidue:
        trace = TraceResidue(
            trace_id=make_id("trace"),
            relation_id=relation_id,
            fragment_id=fragment_id,
            channel=channel,
            intensity=min(max(intensity, 0.0), 1.0),
            carry=min(1.0, 0.40 + intensity * 0.50),
            created_at=created_at,
            last_touched_at=created_at,
        )
        self.traces[trace.trace_id] = trace
        return trace

    def connect(
        self,
        *,
        src_fragment_id: str,
        dst_fragment_id: str,
        kind: AssociationKind,
        weight: float,
        evidence: str,
        now_ts: float,
    ) -> AssociationEdge:
        existing = self.find_edge(src_fragment_id, dst_fragment_id, kind)
        if existing is not None:
            updated = replace(
                existing,
                weight=min(1.0, existing.weight + weight),
                evidence=evidence,
                last_touched_at=now_ts,
            )
            self.associations[existing.edge_id] = updated
            return updated

        edge = AssociationEdge(
            edge_id=make_id("edge"),
            src_fragment_id=src_fragment_id,
            dst_fragment_id=dst_fragment_id,
            kind=kind,
            weight=min(max(weight, 0.0), 1.0),
            evidence=evidence,
            last_touched_at=now_ts,
        )
        self.associations[edge.edge_id] = edge
        return edge

    def find_edge(self, src_fragment_id: str, dst_fragment_id: str, kind: AssociationKind) -> AssociationEdge | None:
        for edge in self.associations.values():
            if edge.kind != kind:
                continue
            if {edge.src_fragment_id, edge.dst_fragment_id} == {src_fragment_id, dst_fragment_id}:
                return edge
        return None

    def fragments_for_relation(self, relation_id: str) -> list[Fragment]:
        return [self.fragments[item] for item in self._relation_fragments.get(relation_id, []) if item in self.fragments]

    def touch_fragment(self, fragment_id: str, *, salience_delta: float, unresolved_delta: float = 0.0) -> Fragment:
        fragment = self.fragments[fragment_id]
        updated = replace(
            fragment,
            salience=min(max(fragment.salience + salience_delta, 0.0), 1.0),
            unresolvedness=min(max(fragment.unresolvedness + unresolved_delta, 0.0), 1.0),
        )
        self.fragments[fragment_id] = updated
        return updated

    def activate_for_relation(self, relation_id: str, signal: TouchSignal, top_k: int = 6) -> list[Fragment]:
        scored: list[tuple[float, Fragment]] = []
        for fragment in self.fragments_for_relation(relation_id):
            score = (
                0.45 * fragment.salience
                + 0.20 * fragment.unresolvedness
                + 0.25 * fragment.touch_signature.overlap(signal)
                + 0.10 * (0.10 if fragment.chapter_ids else 0.0)
            )
            scored.append((score, fragment))

        scored.sort(key=lambda item: item[0], reverse=True)
        activated = [fragment for _, fragment in scored[:top_k]]
        for fragment in activated:
            self.touch_fragment(fragment.fragment_id, salience_delta=0.05)
        return activated

    def select_reweave_candidates(self, relation_id: str | None = None, limit: int = 16) -> list[Fragment]:
        pool = self.fragments.values() if relation_id is None else self.fragments_for_relation(relation_id)
        scored: list[tuple[float, Fragment]] = []
        for fragment in pool:
            trace_weight = sum(
                trace.carry
                for trace in self.traces.values()
                if trace.fragment_id == fragment.fragment_id
            )
            score = 0.45 * fragment.salience + 0.35 * fragment.unresolvedness + 0.20 * min(trace_weight, 1.0)
            scored.append((score, fragment))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [fragment for _, fragment in scored[:limit]]

    def create_chapter(
        self,
        *,
        relation_id: str,
        fragment_ids: tuple[str, ...],
        dominant_channels: tuple[TraceChannel, ...],
        tension: float,
        synopsis: str,
        now_ts: float,
        title: str | None = None,
    ) -> Chapter:
        channel_label = dominant_channels[0] if dominant_channels else "thread"
        chapter = Chapter(
            chapter_id=make_id("chapter"),
            relation_id=relation_id,
            title=title or f"{channel_label}-thread",
            fragment_ids=fragment_ids,
            dominant_channels=dominant_channels,
            tension=min(max(tension, 0.0), 1.0),
            status="forming",
            synopsis=synopsis,
            created_at=now_ts,
            last_rewoven_at=now_ts,
        )
        self.chapters[chapter.chapter_id] = chapter
        for fragment_id in fragment_ids:
            fragment = self.fragments[fragment_id]
            if chapter.chapter_id not in fragment.chapter_ids:
                updated = replace(
                    fragment,
                    chapter_ids=tuple(sorted(set(fragment.chapter_ids + (chapter.chapter_id,)))),
                    salience=min(1.0, fragment.salience + 0.12),
                    unresolvedness=max(0.0, fragment.unresolvedness - 0.06),
                )
                self.fragments[fragment_id] = updated
        return chapter

    def strengthen_cluster_edges(self, fragment_ids: tuple[str, ...], *, now_ts: float) -> list[str]:
        edge_ids: list[str] = []
        for index, src_id in enumerate(fragment_ids):
            for dst_id in fragment_ids[index + 1 :]:
                src = self.fragments[src_id]
                dst = self.fragments[dst_id]
                overlap = src.touch_signature.overlap(dst.touch_signature)
                kind: AssociationKind = "resonance" if overlap >= 0.35 else "contrast"
                edge = self.connect(
                    src_fragment_id=src_id,
                    dst_fragment_id=dst_id,
                    kind=kind,
                    weight=0.20 + overlap * 0.40,
                    evidence="sleep_reweave_cluster",
                    now_ts=now_ts,
                )
                edge_ids.append(edge.edge_id)
        return edge_ids
