from __future__ import annotations

from dataclasses import asdict
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, cast

from aurora.memory.store import MemoryStore
from aurora.persistence.migrations import apply_migrations
from aurora.relation.store import RelationStore
from aurora.runtime.models import (
    AssocKind,
    Association,
    AwakeOutcome,
    BeingState,
    Chapter,
    ChapterRole,
    Fragment,
    Phase,
    PhaseOutcome,
    PhaseTransition,
    RelationMoment,
    RelationState,
    RuntimeState,
    Trace,
    TraceChannel,
)


class SQLitePersistence:
    def __init__(self, data_dir: str | None = None, db_name: str = "aurora.sqlite3") -> None:
        base_dir = Path(data_dir) if data_dir is not None else Path(".aurora")
        base_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = base_dir / db_name
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        apply_migrations(self._connection)

    def load_runtime(self, initial: BeingState) -> tuple[MemoryStore, RelationStore, RuntimeState]:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM runtime_snapshots ORDER BY snapshot_id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return MemoryStore(), RelationStore(), RuntimeState(being=initial, transitions=[])
        payload = json.loads(str(row["payload"]))
        return _decode_payload(payload)

    def persist_awake(
        self,
        outcome: AwakeOutcome,
        state: RuntimeState,
        memory_store: MemoryStore,
        relation_store: RelationStore,
    ) -> None:
        self._persist(state=state, memory_store=memory_store, relation_store=relation_store)

    def persist_phase(
        self,
        outcome: PhaseOutcome,
        state: RuntimeState,
        memory_store: MemoryStore,
        relation_store: RelationStore,
    ) -> None:
        self._persist(state=state, memory_store=memory_store, relation_store=relation_store)

    def turn_count(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM runtime_snapshots ORDER BY snapshot_id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return 0
        payload = json.loads(str(row["payload"]))
        return len(payload.get("relation_moments", []))

    def phase_transition_count(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM runtime_snapshots ORDER BY snapshot_id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return 0
        payload = json.loads(str(row["payload"]))
        return len(payload["transitions"])

    def _persist(
        self, state: RuntimeState, memory_store: MemoryStore, relation_store: RelationStore
    ) -> None:
        payload = _encode_payload(
            state=state, memory_store=memory_store, relation_store=relation_store
        )
        with self._lock:
            with self._connection:
                self._connection.execute(
                    "INSERT INTO runtime_snapshots(payload) VALUES(?)",
                    (json.dumps(payload, ensure_ascii=True, separators=(",", ":")),),
                )


def _encode_payload(
    state: RuntimeState, memory_store: MemoryStore, relation_store: RelationStore
) -> dict[str, Any]:
    return {
        "being": asdict(state.being),
        "transitions": [
            {
                "transition_id": item.transition_id,
                "from_phase": item.from_phase.value,
                "to_phase": item.to_phase.value,
                "reason": item.reason,
                "created_at": item.created_at,
            }
            for item in state.transitions
        ],
        "fragments": [_fragment_dict(item) for item in memory_store.fragments.values()],
        "traces": [_trace_dict(item) for item in memory_store.traces.values()],
        "associations": [_association_dict(item) for item in memory_store.associations.values()],
        "chapters": [_chapter_dict(item) for item in memory_store.chapters.values()],
        "relation_states": [_relation_state_dict(item) for item in relation_store.states.values()],
        "relation_moments": [
            _relation_moment_dict(item)
            for moments in relation_store.moments.values()
            for item in moments
        ],
        "sleep_cycles": memory_store.sleep_cycles,
        "last_reweave_delta": memory_store.last_reweave_delta,
    }


def _decode_payload(payload: dict[str, Any]) -> tuple[MemoryStore, RelationStore, RuntimeState]:
    memory_store = MemoryStore()
    relation_store = RelationStore()
    fragments = cast(list[dict[str, Any]], payload.get("fragments", []))
    traces = cast(list[dict[str, Any]], payload.get("traces", []))
    associations = cast(list[dict[str, Any]], payload.get("associations", []))
    chapters = cast(list[dict[str, Any]], payload.get("chapters", []))
    relation_states = cast(list[dict[str, Any]], payload.get("relation_states", []))
    relation_moments = cast(list[dict[str, Any]], payload.get("relation_moments", []))
    being_raw = cast(dict[str, Any], payload.get("being", {}))
    transitions_raw = cast(list[dict[str, Any]], payload.get("transitions", []))

    for raw in fragments:
        item = Fragment(
            fragment_id=raw["fragment_id"],
            relation_id=raw["relation_id"],
            turn_id=raw["turn_id"],
            surface=raw["surface"],
            tags=tuple(raw["tags"]),
            vividness=raw["vividness"],
            salience=raw["salience"],
            unresolvedness=raw["unresolvedness"],
            chapter_ids=tuple(raw["chapter_ids"]),
            created_at=raw["created_at"],
            last_touched_at=raw["last_touched_at"],
            activation_count=raw["activation_count"],
        )
        memory_store.add_fragment(item)

    for raw in traces:
        memory_store.add_trace(
            Trace(
                trace_id=raw["trace_id"],
                relation_id=raw["relation_id"],
                fragment_id=raw["fragment_id"],
                channel=TraceChannel(raw["channel"]),
                intensity=raw["intensity"],
                decay_rate=raw["decay_rate"],
                created_at=raw["created_at"],
                last_touched_at=raw["last_touched_at"],
            )
        )

    for raw in associations:
        memory_store.add_association(
            Association(
                edge_id=raw["edge_id"],
                src_fragment_id=raw["src_fragment_id"],
                dst_fragment_id=raw["dst_fragment_id"],
                kind=AssocKind(raw["kind"]),
                weight=raw["weight"],
                evidence=tuple(raw["evidence"]),
                created_at=raw["created_at"],
                last_touched_at=raw["last_touched_at"],
            )
        )

    for raw in chapters:
        memory_store.add_chapter(
            Chapter(
                chapter_id=raw["chapter_id"],
                relation_id=raw["relation_id"],
                title=raw["title"],
                motif=raw["motif"],
                fragment_ids=tuple(raw["fragment_ids"]),
                roles={key: ChapterRole(value) for key, value in raw["roles"].items()},
                tension=raw["tension"],
                coherence=raw["coherence"],
                created_at=raw["created_at"],
                last_rewoven_at=raw["last_rewoven_at"],
            )
        )

    for raw in relation_states:
        relation_store.states[raw["relation_id"]] = RelationState(
            relation_id=raw["relation_id"],
            trust=raw["trust"],
            reciprocity=raw["reciprocity"],
            boundary_tension=raw["boundary_tension"],
            repairability=raw["repairability"],
            distance=raw["distance"],
            shared_chapters=set(raw["shared_chapters"]),
            last_contact_at=raw["last_contact_at"],
        )

    for raw in relation_moments:
        moment = RelationMoment(
            moment_id=raw["moment_id"],
            relation_id=raw["relation_id"],
            user_turn_id=raw["user_turn_id"],
            aurora_turn_id=raw["aurora_turn_id"],
            user_channels=tuple(TraceChannel(item) for item in raw["user_channels"]),
            aurora_move=raw["aurora_move"],
            boundary_crossed=raw["boundary_crossed"],
            repair_attempted=raw["repair_attempted"],
            summary=raw["summary"],
            created_at=raw["created_at"],
        )
        relation_store.moments[moment.relation_id].append(moment)

    memory_store.sleep_cycles = int(cast(int | float | str, payload.get("sleep_cycles", 0)))
    memory_store.last_reweave_delta = float(
        cast(int | float | str, payload.get("last_reweave_delta", 0.0))
    )
    being = BeingState(
        phase=Phase(being_raw["phase"]),
        continuity_pressure=being_raw["continuity_pressure"],
        sleep_pressure=being_raw["sleep_pressure"],
        coherence_pressure=being_raw["coherence_pressure"],
        softness=being_raw["softness"],
        boundary_tension=being_raw["boundary_tension"],
        self_vector=dict(being_raw["self_vector"]),
        world_vector=dict(being_raw["world_vector"]),
        recent_chapter_bias=tuple(being_raw["recent_chapter_bias"]),
        active_relation_id=being_raw["active_relation_id"],
    )
    transitions = [
        PhaseTransition(
            transition_id=item["transition_id"],
            from_phase=Phase(item["from_phase"]),
            to_phase=Phase(item["to_phase"]),
            reason=item["reason"],
            created_at=item["created_at"],
        )
        for item in transitions_raw
    ]
    return memory_store, relation_store, RuntimeState(being=being, transitions=transitions)


def _fragment_dict(item: Fragment) -> dict[str, object]:
    return {
        "fragment_id": item.fragment_id,
        "relation_id": item.relation_id,
        "turn_id": item.turn_id,
        "surface": item.surface,
        "tags": list(item.tags),
        "vividness": item.vividness,
        "salience": item.salience,
        "unresolvedness": item.unresolvedness,
        "chapter_ids": list(item.chapter_ids),
        "created_at": item.created_at,
        "last_touched_at": item.last_touched_at,
        "activation_count": item.activation_count,
    }


def _trace_dict(item: Trace) -> dict[str, object]:
    return {
        "trace_id": item.trace_id,
        "relation_id": item.relation_id,
        "fragment_id": item.fragment_id,
        "channel": item.channel.value,
        "intensity": item.intensity,
        "decay_rate": item.decay_rate,
        "created_at": item.created_at,
        "last_touched_at": item.last_touched_at,
    }


def _association_dict(item: Association) -> dict[str, object]:
    return {
        "edge_id": item.edge_id,
        "src_fragment_id": item.src_fragment_id,
        "dst_fragment_id": item.dst_fragment_id,
        "kind": item.kind.value,
        "weight": item.weight,
        "evidence": list(item.evidence),
        "created_at": item.created_at,
        "last_touched_at": item.last_touched_at,
    }


def _chapter_dict(item: Chapter) -> dict[str, object]:
    return {
        "chapter_id": item.chapter_id,
        "relation_id": item.relation_id,
        "title": item.title,
        "motif": item.motif,
        "fragment_ids": list(item.fragment_ids),
        "roles": {key: value.value for key, value in item.roles.items()},
        "tension": item.tension,
        "coherence": item.coherence,
        "created_at": item.created_at,
        "last_rewoven_at": item.last_rewoven_at,
    }


def _relation_state_dict(item: RelationState) -> dict[str, object]:
    return {
        "relation_id": item.relation_id,
        "trust": item.trust,
        "reciprocity": item.reciprocity,
        "boundary_tension": item.boundary_tension,
        "repairability": item.repairability,
        "distance": item.distance,
        "shared_chapters": sorted(item.shared_chapters),
        "last_contact_at": item.last_contact_at,
    }


def _relation_moment_dict(item: RelationMoment) -> dict[str, object]:
    return {
        "moment_id": item.moment_id,
        "relation_id": item.relation_id,
        "user_turn_id": item.user_turn_id,
        "aurora_turn_id": item.aurora_turn_id,
        "user_channels": [channel.value for channel in item.user_channels],
        "aurora_move": item.aurora_move,
        "boundary_crossed": item.boundary_crossed,
        "repair_attempted": item.repair_attempted,
        "summary": item.summary,
        "created_at": item.created_at,
    }
