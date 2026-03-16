from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from aurora.memory.recall import SALIENCE_FLOOR

if TYPE_CHECKING:
    from aurora.memory.store import MemoryStore

UNRESOLVED_FLOOR = 0.06
STALENESS_HOURS = 168.0
ASSOCIATION_WEIGHT_FLOOR = 0.10


@dataclass(frozen=True, slots=True)
class SedimentResult:
    removed_fragment_ids: tuple[str, ...]
    removed_trace_ids: tuple[str, ...]
    removed_association_ids: tuple[str, ...]
    removed_thread_ids: tuple[str, ...]
    removed_knot_ids: tuple[str, ...]


def _is_sediment_candidate(fragment: "Fragment", now_ts: float) -> bool:
    hours_stale = (now_ts - fragment.last_touched_at) / 3600.0
    below_floor = (
        fragment.salience < SALIENCE_FLOOR
        and fragment.unresolvedness < UNRESOLVED_FLOOR
    )
    return below_floor or hours_stale >= STALENESS_HOURS


def sediment(store: MemoryStore, now_ts: float) -> SedimentResult:
    candidate_fids = {
        fid for fid, f in store.fragments.items() if _is_sediment_candidate(f, now_ts)
    }

    remove_thread_ids: list[str] = []
    for thread in list(store.threads.values()):
        if set(thread.fragment_ids) <= candidate_fids:
            remove_thread_ids.append(thread.thread_id)
    remove_knot_ids: list[str] = []
    for knot in list(store.knots.values()):
        if set(knot.fragment_ids) <= candidate_fids:
            remove_knot_ids.append(knot.knot_id)

    for thread_id in remove_thread_ids:
        store.remove_thread(thread_id)
    for knot_id in remove_knot_ids:
        store.remove_knot(knot_id)

    active_fids: set[str] = set()
    for thread in store.threads.values():
        active_fids.update(thread.fragment_ids)
    for knot in store.knots.values():
        active_fids.update(knot.fragment_ids)

    remove_fids = [fid for fid in candidate_fids if fid not in active_fids]

    remove_tids: list[str] = []
    for fid in remove_fids:
        for tid in list(store.fragment_traces.get(fid, ())):
            remove_tids.append(tid)

    removed_fid_set = set(remove_fids)
    remove_eids: list[str] = []
    for eid, edge in list(store.associations.items()):
        both_gone = (
            edge.src_fragment_id in removed_fid_set
            and edge.dst_fragment_id in removed_fid_set
        )
        one_gone = (
            edge.src_fragment_id in removed_fid_set
            or edge.dst_fragment_id in removed_fid_set
        )
        if both_gone or (one_gone and edge.weight < ASSOCIATION_WEIGHT_FLOOR):
            remove_eids.append(eid)

    for tid in remove_tids:
        store.remove_trace(tid)
    for eid in remove_eids:
        store.remove_association(eid)
    for fid in remove_fids:
        store.remove_fragment(fid)

    return SedimentResult(
        removed_fragment_ids=tuple(remove_fids),
        removed_trace_ids=tuple(remove_tids),
        removed_association_ids=tuple(remove_eids),
        removed_thread_ids=tuple(remove_thread_ids),
        removed_knot_ids=tuple(remove_knot_ids),
    )
