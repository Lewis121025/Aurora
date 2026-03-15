from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Iterable
from uuid import uuid4

from aurora.memory.affinity import (
    cluster_dominant_channels,
    cluster_keyword_overlap,
    cluster_trace_overlap,
    fragment_affinity,
    neighbor_fragment_ids,
    region_edge_density,
    structural_pressure,
)
from aurora.memory.fragment import Fragment
from aurora.memory.knot import Knot
from aurora.memory.reweave import NarrativeRegion, SleepMutation
from aurora.memory.thread import Thread
from aurora.runtime.contracts import AssocKind, TraceChannel, clamp

if TYPE_CHECKING:
    from aurora.memory.store import MemoryStore
    from aurora.relation.formation import RelationFormation

# Candidate selection weights
CANDIDATE_SALIENCE_WEIGHT = 0.36
CANDIDATE_UNRESOLVEDNESS_WEIGHT = 0.28
CANDIDATE_ACTIVATION_WEIGHT = 0.22
CANDIDATE_STRUCTURAL_WEIGHT = 0.08
CANDIDATE_THREAD_WEIGHT = 0.03
CANDIDATE_KNOT_WEIGHT = 0.03
CANDIDATE_ACTIVATION_CAP = 5.0

# Region seed priority weights
SEED_SALIENCE_WEIGHT = 0.32
SEED_UNRESOLVEDNESS_WEIGHT = 0.30
SEED_ACTIVATION_WEIGHT = 0.16
SEED_STRUCTURAL_WEIGHT = 0.12
SEED_FORMATION_THREAD_BONUS = 0.08
SEED_FORMATION_KNOT_BONUS = 0.12

# Region affinity thresholds
REGION_AFFINITY_THRESHOLD = 0.34
REGION_OVERLAP_THRESHOLD = 0.75

# Region support weights
SUPPORT_EDGE_DENSITY_WEIGHT = 0.34
SUPPORT_THREAD_PRESENCE_WEIGHT = 0.24
SUPPORT_KNOT_PRESENCE_WEIGHT = 0.24
SUPPORT_FORMATION_WEIGHT = 0.18

# Region coherence
COHERENCE_BASE = 0.22
COHERENCE_EDGE_DENSITY_WEIGHT = 0.18
COHERENCE_FORMATION_THREAD_BONUS = 0.08

# Knot formation threshold
KNOT_BASE_THRESHOLD = 0.56
KNOT_BOUNDARY_DISCOUNT = 0.03
KNOT_CHANNEL_DISCOUNT = 0.02

# Association strengthening weights
ASSOC_BASE_WEIGHT = 0.46
ASSOC_AFFINITY_WEIGHT = 0.24
ASSOC_COHERENCE_WEIGHT = 0.16
ASSOC_SUPPORT_WEIGHT = 0.12

# Fragment softening during reweave
SOFTEN_SALIENCE_BASE = 0.05
SOFTEN_SALIENCE_SUPPORT_FACTOR = 0.03
SOFTEN_UNRESOLVED_BASE = 0.05
SOFTEN_UNRESOLVED_COHERENCE_FACTOR = 0.05


def reweave(
    store: MemoryStore,
    relation_formations: dict[str, RelationFormation],
    now_ts: float,
    pending_relations: tuple[str, ...] | None = None,
) -> SleepMutation:
    relation_ids = pending_relations or tuple(sorted(store.relation_fragments.keys()))

    created_thread_ids: list[str] = []
    updated_thread_ids: list[str] = []
    created_knot_ids: list[str] = []
    updated_knot_ids: list[str] = []
    strengthened_edge_ids: list[str] = []
    softened_fragment_ids: set[str] = set()
    affected_relation_ids: list[str] = []
    recall_bias: dict[str, tuple[str, ...]] = {}

    for relation_id in relation_ids:
        formation = relation_formations.get(relation_id)
        candidates = _select_candidates(store, relation_id)
        regions = _build_regions(store, relation_id, candidates, formation)
        if not regions:
            continue

        affected_relation_ids.append(relation_id)
        relation_thread_ids: list[str] = []
        relation_knot_ids: list[str] = []

        for region in regions[:3]:
            cluster = [store.fragments[fid] for fid in region.fragment_ids]
            frag_ids = tuple(f.fragment_id for f in cluster)

            existing_thread = store.find_matching_thread(relation_id, frag_ids)
            if existing_thread is not None:
                thread = _update_thread(
                    existing_thread, cluster, region.dominant_channels,
                    region.tension, region.coherence, now_ts,
                )
                store.update_thread(thread)
                updated_thread_ids.append(thread.thread_id)
            else:
                thread = _build_thread(
                    relation_id, cluster, region.dominant_channels,
                    region.tension, region.coherence, now_ts,
                )
                store.add_thread(thread)
                created_thread_ids.append(thread.thread_id)
            relation_thread_ids.append(thread.thread_id)

            knot: Knot | None = None
            if _should_form_knot(region, formation):
                existing_knot = store.find_matching_knot(relation_id, frag_ids)
                if existing_knot is not None:
                    knot = _update_knot(
                        existing_knot, cluster, region.dominant_channels,
                        region.tension, now_ts,
                    )
                    store.update_knot(knot)
                    updated_knot_ids.append(knot.knot_id)
                else:
                    knot = _build_knot(
                        relation_id, cluster, region.dominant_channels, region.tension, now_ts,
                    )
                    store.add_knot(knot)
                    created_knot_ids.append(knot.knot_id)
                relation_knot_ids.append(knot.knot_id)

            for fragment in cluster:
                softened_fragment_ids.add(fragment.fragment_id)
                store.fragments[fragment.fragment_id] = store.fragments[
                    fragment.fragment_id
                ].touched(
                    at=now_ts,
                    delta_salience=SOFTEN_SALIENCE_BASE + SOFTEN_SALIENCE_SUPPORT_FACTOR * region.support,
                    delta_unresolved=-(SOFTEN_UNRESOLVED_BASE + SOFTEN_UNRESOLVED_COHERENCE_FACTOR * region.coherence),
                )

            edge_kind = AssocKind.KNOT if knot is not None else AssocKind.THREAD
            evidence_token = knot.knot_id if knot is not None else thread.thread_id
            for left, right in combinations(cluster, 2):
                edge = store.strengthen_association(
                    src_fragment_id=left.fragment_id,
                    dst_fragment_id=right.fragment_id,
                    kind=edge_kind,
                    weight=(
                        ASSOC_BASE_WEIGHT
                        + ASSOC_AFFINITY_WEIGHT * fragment_affinity(store, left, right)
                        + ASSOC_COHERENCE_WEIGHT * region.coherence
                        + ASSOC_SUPPORT_WEIGHT * region.support
                    ),
                    evidence=(evidence_token,),
                    now_ts=now_ts,
                )
                strengthened_edge_ids.append(edge.edge_id)

        recall_bias[relation_id] = tuple(relation_thread_ids[-4:])
        if formation is not None:
            formation.absorb_sleep(
                thread_ids=tuple(relation_thread_ids),
                knot_ids=tuple(relation_knot_ids),
                now_ts=now_ts,
            )

    all_thread_ids = created_thread_ids + updated_thread_ids
    all_knot_ids = created_knot_ids + updated_knot_ids
    if all_thread_ids or all_knot_ids:
        store.sleep_cycles += 1
        store.last_sleep_at = now_ts

    return SleepMutation(
        created_thread_ids=tuple(all_thread_ids),
        created_knot_ids=tuple(all_knot_ids),
        strengthened_edge_ids=tuple(strengthened_edge_ids),
        softened_fragment_ids=tuple(sorted(softened_fragment_ids)),
        affected_relation_ids=tuple(affected_relation_ids),
        recall_bias=recall_bias,
    )


def _select_candidates(
    store: MemoryStore, relation_id: str, top_k: int = 24
) -> tuple[Fragment, ...]:
    ranked = sorted(
        store.fragments_for_relation(relation_id),
        key=lambda item: (
            CANDIDATE_SALIENCE_WEIGHT * item.salience
            + CANDIDATE_UNRESOLVEDNESS_WEIGHT * item.unresolvedness
            + CANDIDATE_ACTIVATION_WEIGHT * min(item.activation_count / CANDIDATE_ACTIVATION_CAP, 1.0)
            + CANDIDATE_STRUCTURAL_WEIGHT * structural_pressure(store, item)
            + CANDIDATE_THREAD_WEIGHT * min(len(item.thread_ids), 2)
            + CANDIDATE_KNOT_WEIGHT * min(len(item.knot_ids), 2)
        ),
        reverse=True,
    )
    return tuple(ranked[:top_k])


def _build_regions(
    store: MemoryStore,
    relation_id: str,
    candidates: Iterable[Fragment],
    formation: RelationFormation | None,
) -> tuple[NarrativeRegion, ...]:
    ordered = sorted(
        candidates,
        key=lambda item: _seed_priority(store, item, formation),
        reverse=True,
    )
    regions: list[NarrativeRegion] = []
    for seed in ordered[:8]:
        fragment_ids = _expand_region(store, seed, ordered, formation)
        if len(fragment_ids) < 2:
            continue
        region = _materialize_region(store, relation_id, seed.fragment_id, fragment_ids, formation)
        if any(
            _region_overlap(region.fragment_ids, existing.fragment_ids) >= REGION_OVERLAP_THRESHOLD
            for existing in regions
        ):
            continue
        regions.append(region)
    return tuple(sorted(regions, key=lambda item: (item.support, item.tension), reverse=True))


def _seed_priority(
    store: MemoryStore,
    fragment: Fragment,
    formation: RelationFormation | None,
) -> float:
    formation_thread_bonus = 0.0
    formation_knot_bonus = 0.0
    if formation is not None:
        formation_thread_bonus = (
            SEED_FORMATION_THREAD_BONUS if set(fragment.thread_ids) & formation.thread_ids else 0.0
        )
        formation_knot_bonus = (
            SEED_FORMATION_KNOT_BONUS if set(fragment.knot_ids) & formation.knot_ids else 0.0
        )
    return (
        SEED_SALIENCE_WEIGHT * fragment.salience
        + SEED_UNRESOLVEDNESS_WEIGHT * fragment.unresolvedness
        + SEED_ACTIVATION_WEIGHT * min(fragment.activation_count / CANDIDATE_ACTIVATION_CAP, 1.0)
        + SEED_STRUCTURAL_WEIGHT * structural_pressure(store, fragment)
        + formation_thread_bonus
        + formation_knot_bonus
    )


def _expand_region(
    store: MemoryStore,
    seed: Fragment,
    candidates: Iterable[Fragment],
    formation: RelationFormation | None,
) -> tuple[str, ...]:
    region_ids: list[str] = [seed.fragment_id]
    for nid in neighbor_fragment_ids(store, seed.fragment_id):
        if nid in store.fragments and nid not in region_ids:
            region_ids.append(nid)
    for linked_id in _linked_fragment_ids(store, seed):
        if linked_id in store.fragments and linked_id not in region_ids:
            region_ids.append(linked_id)

    scored: list[tuple[float, str]] = []
    for fragment in candidates:
        if fragment.fragment_id in region_ids:
            continue
        support = _region_affinity(store, seed, fragment, formation)
        if support >= REGION_AFFINITY_THRESHOLD:
            scored.append((support, fragment.fragment_id))
    scored.sort(reverse=True)
    for _, fragment_id in scored[:3]:
        if fragment_id not in region_ids:
            region_ids.append(fragment_id)
    ordered = sorted(
        (store.fragments[fid] for fid in region_ids),
        key=lambda item: item.created_at,
    )
    return tuple(f.fragment_id for f in ordered)


def _linked_fragment_ids(store: MemoryStore, fragment: Fragment) -> tuple[str, ...]:
    linked: list[str] = []
    for thread_id in fragment.thread_ids:
        thread = store.threads.get(thread_id)
        if thread is None:
            continue
        for fid in thread.fragment_ids:
            if fid != fragment.fragment_id and fid not in linked:
                linked.append(fid)
    for knot_id in fragment.knot_ids:
        knot = store.knots.get(knot_id)
        if knot is None:
            continue
        for fid in knot.fragment_ids:
            if fid != fragment.fragment_id and fid not in linked:
                linked.append(fid)
    return tuple(linked)


def _region_affinity(
    store: MemoryStore,
    seed: Fragment,
    fragment: Fragment,
    formation: RelationFormation | None,
) -> float:
    structural = (
        0.12 if fragment.fragment_id in neighbor_fragment_ids(store, seed.fragment_id) else 0.0
    )
    thread_overlap = 0.14 if set(seed.thread_ids) & set(fragment.thread_ids) else 0.0
    knot_overlap = 0.18 if set(seed.knot_ids) & set(fragment.knot_ids) else 0.0
    formation_overlap = 0.0
    if formation is not None and (
        set(fragment.thread_ids) & formation.thread_ids
        or set(fragment.knot_ids) & formation.knot_ids
    ):
        formation_overlap = 0.08
    return (
        fragment_affinity(store, seed, fragment)
        + structural
        + thread_overlap
        + knot_overlap
        + formation_overlap
    )


def _materialize_region(
    store: MemoryStore,
    relation_id: str,
    anchor_fragment_id: str,
    fragment_ids: tuple[str, ...],
    formation: RelationFormation | None,
) -> NarrativeRegion:
    fragments = [store.fragments[fid] for fid in fragment_ids]
    dominant = cluster_dominant_channels(store, fragments)
    support = _region_support(store, fragments, formation)
    tension = _region_tension(store, fragments, formation)
    coherence = _region_coherence(store, fragments, formation)
    return NarrativeRegion(
        relation_id=relation_id,
        anchor_fragment_id=anchor_fragment_id,
        fragment_ids=fragment_ids,
        dominant_channels=dominant,
        tension=round(clamp(tension), 4),
        coherence=round(clamp(coherence), 4),
        support=round(clamp(support), 4),
    )


def _region_support(
    store: MemoryStore,
    fragments: list[Fragment],
    formation: RelationFormation | None,
) -> float:
    edge_density = region_edge_density(
        store, tuple(f.fragment_id for f in fragments)
    )
    thread_presence = sum(1 for f in fragments if f.thread_ids) / max(1, len(fragments))
    knot_presence = sum(1 for f in fragments if f.knot_ids) / max(1, len(fragments))
    formation_bonus = 0.0
    if formation is not None:
        formation_hits = sum(
            1 for f in fragments
            if set(f.thread_ids) & formation.thread_ids or set(f.knot_ids) & formation.knot_ids
        )
        formation_bonus = formation_hits / max(1, len(fragments))
    return (
        SUPPORT_EDGE_DENSITY_WEIGHT * edge_density
        + SUPPORT_THREAD_PRESENCE_WEIGHT * thread_presence
        + SUPPORT_KNOT_PRESENCE_WEIGHT * knot_presence
        + SUPPORT_FORMATION_WEIGHT * formation_bonus
    )


def _region_tension(
    store: MemoryStore,
    fragments: list[Fragment],
    formation: RelationFormation | None,
) -> float:
    base = sum(f.unresolvedness for f in fragments) / len(fragments)
    channels = set(cluster_dominant_channels(store, fragments))
    if TraceChannel.HURT in channels:
        base += 0.08
    if TraceChannel.BOUNDARY in channels:
        base += 0.08
    if formation is not None and formation.boundary_events > formation.repair_events:
        base += 0.06
    return base


def _region_coherence(
    store: MemoryStore,
    fragments: list[Fragment],
    formation: RelationFormation | None,
) -> float:
    coherence = (
        COHERENCE_BASE
        + cluster_keyword_overlap(fragments)
        + cluster_trace_overlap(store, fragments)
    )
    coherence += COHERENCE_EDGE_DENSITY_WEIGHT * region_edge_density(
        store, tuple(f.fragment_id for f in fragments)
    )
    if formation is not None:
        thread_hit = any(set(f.thread_ids) & formation.thread_ids for f in fragments)
        if thread_hit:
            coherence += COHERENCE_FORMATION_THREAD_BONUS
    return coherence


def _region_overlap(
    left_fragment_ids: tuple[str, ...],
    right_fragment_ids: tuple[str, ...],
) -> float:
    left = set(left_fragment_ids)
    right = set(right_fragment_ids)
    return len(left & right) / max(1, len(left | right))


def _should_form_knot(
    region: NarrativeRegion,
    formation: RelationFormation | None,
) -> bool:
    threshold = KNOT_BASE_THRESHOLD
    if formation is not None and formation.boundary_events > formation.repair_events:
        threshold -= KNOT_BOUNDARY_DISCOUNT
    if (
        TraceChannel.BOUNDARY in region.dominant_channels
        or TraceChannel.HURT in region.dominant_channels
    ):
        threshold -= KNOT_CHANNEL_DISCOUNT
    return region.tension >= threshold


def _update_thread(
    existing: Thread,
    cluster: list[Fragment],
    dominant_channels: tuple[TraceChannel, ...],
    tension: float,
    coherence: float,
    now_ts: float,
) -> Thread:
    return Thread(
        thread_id=existing.thread_id,
        relation_id=existing.relation_id,
        fragment_ids=tuple(f.fragment_id for f in cluster),
        dominant_channels=dominant_channels,
        tension=round(clamp(tension), 4),
        coherence=round(clamp(coherence), 4),
        created_at=existing.created_at,
        last_rewoven_at=now_ts,
    )


def _build_thread(
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
        fragment_ids=tuple(f.fragment_id for f in cluster),
        dominant_channels=dominant_channels,
        tension=round(clamp(tension), 4),
        coherence=round(clamp(coherence), 4),
        created_at=now_ts,
        last_rewoven_at=now_ts,
    )


def _update_knot(
    existing: Knot,
    cluster: list[Fragment],
    dominant_channels: tuple[TraceChannel, ...],
    intensity: float,
    now_ts: float,
) -> Knot:
    return Knot(
        knot_id=existing.knot_id,
        relation_id=existing.relation_id,
        fragment_ids=tuple(f.fragment_id for f in cluster),
        dominant_channels=dominant_channels,
        intensity=round(clamp(intensity), 4),
        resolved=existing.resolved,
        created_at=existing.created_at,
        last_rewoven_at=now_ts,
    )


def _build_knot(
    relation_id: str,
    cluster: list[Fragment],
    dominant_channels: tuple[TraceChannel, ...],
    intensity: float,
    now_ts: float,
) -> Knot:
    return Knot(
        knot_id=f"knot_{uuid4().hex[:12]}",
        relation_id=relation_id,
        fragment_ids=tuple(f.fragment_id for f in cluster),
        dominant_channels=dominant_channels,
        intensity=round(clamp(intensity), 4),
        resolved=False,
        created_at=now_ts,
        last_rewoven_at=now_ts,
    )
