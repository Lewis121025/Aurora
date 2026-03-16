from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, cast

from aurora.being.metabolic_state import MetabolicState
from aurora.being.orientation import Orientation
from aurora.memory.association import Association
from aurora.memory.fragment import Fragment
from aurora.memory.knot import Knot
from aurora.memory.store import MemoryStore
from aurora.memory.thread import Thread
from aurora.memory.trace import Trace
from aurora.persistence.migrations import apply_migrations
from aurora.phases.awake import AwakeOutcome
from aurora.phases.outcomes import PhaseOutcome
from aurora.relation.formation import RelationFormation
from aurora.relation.moment import RelationMoment
from aurora.relation.store import RelationStore
from aurora.runtime.contracts import AssocKind, Phase, PhaseTransition, TraceChannel, Turn
from aurora.runtime.state import RuntimeState


class SQLitePersistence:
    def __init__(self, data_dir: str | None = None, db_name: str = "aurora.sqlite3") -> None:
        base_dir = Path(data_dir) if data_dir is not None else Path(".aurora")
        base_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = base_dir / db_name
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()
        apply_migrations(self._connection)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        try:
            self._connection.execute("SELECT id, phase, sleep_need FROM metabolic_state LIMIT 1")
            self._connection.execute("SELECT id, self_evidence FROM orientation_state LIMIT 1")
            self._connection.execute("SELECT thread_id FROM threads LIMIT 1")
        except sqlite3.OperationalError as exc:
            raise RuntimeError(
                f"Incompatible database schema at {self.db_path}. "
                "Delete the database file to reset, or migrate manually."
            ) from exc

    def load_runtime(
        self,
        initial: RuntimeState,
    ) -> tuple[MemoryStore, RelationStore, RuntimeState]:
        memory_store = MemoryStore()
        relation_store = RelationStore()

        with self._lock:
            orientation_row = self._connection.execute(
                "SELECT * FROM orientation_state WHERE id = 1"
            ).fetchone()
            metabolic_row = self._connection.execute(
                "SELECT * FROM metabolic_state WHERE id = 1"
            ).fetchone()

            fragment_rows = self._connection.execute("SELECT * FROM fragments").fetchall()
            trace_rows = self._connection.execute("SELECT * FROM traces").fetchall()
            association_rows = self._connection.execute("SELECT * FROM associations").fetchall()
            thread_rows = self._connection.execute("SELECT * FROM threads").fetchall()
            knot_rows = self._connection.execute("SELECT * FROM knots").fetchall()
            formation_rows = self._connection.execute(
                "SELECT * FROM relation_formations"
            ).fetchall()
            moment_rows = self._connection.execute("SELECT * FROM relation_moments").fetchall()
            transition_rows = self._connection.execute(
                "SELECT * FROM phase_events ORDER BY created_at ASC"
            ).fetchall()

        if orientation_row is None or metabolic_row is None:
            return MemoryStore(), RelationStore(), initial

        for row in fragment_rows:
            memory_store.add_fragment(
                Fragment(
                    fragment_id=str(row["fragment_id"]),
                    relation_id=str(row["relation_id"]),
                    turn_id=str(row["turn_id"]) if row["turn_id"] is not None else None,
                    surface=str(row["surface"]),
                    tags=_load_tuple_str(row["tags"]),
                    vividness=float(row["vividness"]),
                    salience=float(row["salience"]),
                    unresolvedness=float(row["unresolvedness"]),
                    thread_ids=_load_tuple_str(row["thread_ids"]),
                    knot_ids=_load_tuple_str(row["knot_ids"]),
                    created_at=float(row["created_at"]),
                    last_touched_at=float(row["last_touched_at"]),
                    activation_count=int(row["activation_count"]),
                )
            )

        for row in trace_rows:
            memory_store.add_trace(
                Trace(
                    trace_id=str(row["trace_id"]),
                    relation_id=str(row["relation_id"]),
                    fragment_id=str(row["fragment_id"]),
                    channel=TraceChannel(str(row["channel"])),
                    intensity=float(row["intensity"]),
                    carry=float(row["carry"]),
                    created_at=float(row["created_at"]),
                    last_touched_at=float(row["last_touched_at"]),
                )
            )

        for row in association_rows:
            memory_store.add_association(
                Association(
                    edge_id=str(row["edge_id"]),
                    src_fragment_id=str(row["src_fragment_id"]),
                    dst_fragment_id=str(row["dst_fragment_id"]),
                    kind=AssocKind(str(row["kind"])),
                    weight=float(row["weight"]),
                    evidence=_load_tuple_str(row["evidence"]),
                    created_at=float(row["created_at"]),
                    last_touched_at=float(row["last_touched_at"]),
                )
            )

        for row in thread_rows:
            memory_store.add_thread(
                Thread(
                    thread_id=str(row["thread_id"]),
                    relation_id=str(row["relation_id"]),
                    fragment_ids=_load_tuple_str(row["fragment_ids"]),
                    dominant_channels=tuple(
                        TraceChannel(item) for item in _load_tuple_str(row["dominant_channels"])
                    ),
                    tension=float(row["tension"]),
                    coherence=float(row["coherence"]),
                    created_at=float(row["created_at"]),
                    last_rewoven_at=float(row["last_rewoven_at"]),
                )
            )

        for row in knot_rows:
            memory_store.add_knot(
                Knot(
                    knot_id=str(row["knot_id"]),
                    relation_id=str(row["relation_id"]),
                    fragment_ids=_load_tuple_str(row["fragment_ids"]),
                    dominant_channels=tuple(
                        TraceChannel(item) for item in _load_tuple_str(row["dominant_channels"])
                    ),
                    intensity=float(row["intensity"]),
                    resolved=bool(int(row["resolved"])),
                    created_at=float(row["created_at"]),
                    last_rewoven_at=float(row["last_rewoven_at"]),
                )
            )

        for row in formation_rows:
            relation_store.formations[str(row["relation_id"])] = RelationFormation(
                relation_id=str(row["relation_id"]),
                thread_ids=set(_load_tuple_str(row["thread_ids"])),
                knot_ids=set(_load_tuple_str(row["knot_ids"])),
                boundary_events=int(row["boundary_events"]),
                repair_events=int(row["repair_events"]),
                resonance_events=int(row["resonance_events"]),
                last_contact_at=float(row["last_contact_at"]),
            )

        for row in moment_rows:
            moment = RelationMoment(
                moment_id=str(row["moment_id"]),
                relation_id=str(row["relation_id"]),
                user_turn_id=str(row["user_turn_id"]),
                aurora_turn_id=str(row["aurora_turn_id"]) if row["aurora_turn_id"] else None,
                user_channels=tuple(
                    TraceChannel(item) for item in _load_tuple_str(row["user_channels"])
                ),
                aurora_move=cast(Any, str(row["aurora_move"])),
                boundary_event=bool(int(row["boundary_event"])),
                repair_event=bool(int(row["repair_event"])),
                summary=str(row["summary"]),
                created_at=float(row["created_at"]),
            )
            relation_store.moments[moment.relation_id].append(moment)

        for relation_id, moments in relation_store.moments.items():
            if relation_id in relation_store.formations:
                continue
            formation = relation_store.formation_for(relation_id)
            for moment in moments:
                formation.register_moment(moment)

        transitions = [
            PhaseTransition(
                transition_id=str(row["transition_id"]),
                from_phase=Phase(str(row["from_phase"])),
                to_phase=Phase(str(row["to_phase"])),
                reason=str(row["reason"]),
                created_at=float(row["created_at"]),
            )
            for row in transition_rows
        ]
        memory_store.sleep_cycles = sum(1 for item in transitions if item.to_phase is Phase.SLEEP)
        sleep_transitions = [
            item.created_at for item in transitions if item.to_phase is Phase.SLEEP
        ]
        memory_store.last_sleep_at = max(sleep_transitions) if sleep_transitions else 0.0

        orientation = Orientation(
            self_evidence=_load_evidence_map(orientation_row["self_evidence"]),
            world_evidence=_load_evidence_map(orientation_row["world_evidence"]),
            relation_evidence=_load_evidence_map(orientation_row["relation_evidence"]),
            anchor_thread_ids=_load_tuple_str(orientation_row["anchor_thread_ids"]),
            active_knot_ids=_load_tuple_str(orientation_row["active_knot_ids"]),
            last_updated_at=float(orientation_row["last_updated_at"]),
        )
        metabolic = MetabolicState(
            phase=Phase(str(metabolic_row["phase"])),
            sleep_need=float(metabolic_row["sleep_need"]),
            active_relation_ids=_load_tuple_str(metabolic_row["active_relation_ids"]),
            active_knot_ids=_load_tuple_str(metabolic_row["active_knot_ids"]),
            pending_sleep_relation_ids=_load_tuple_str(metabolic_row["pending_sleep_relation_ids"]),
            last_transition_at=float(metabolic_row["last_transition_at"]),
        )
        return (
            memory_store,
            relation_store,
            RuntimeState(
                orientation=orientation,
                metabolic=metabolic,
                transitions=transitions,
            ),
        )

    def persist_awake(
        self,
        outcome: AwakeOutcome,
        state: RuntimeState,
        memory_store: MemoryStore,
        relation_store: RelationStore,
    ) -> None:
        with self._lock:
            with self._connection:
                self._insert_turn(outcome.user_turn)
                self._insert_turn(outcome.aurora_turn)
                if outcome.transition is not None:
                    self._insert_transition(outcome.transition)
                self._persist_runtime_tables(
                    state=state,
                    memory_store=memory_store,
                    relation_store=relation_store,
                )

    def persist_phase(
        self,
        outcome: PhaseOutcome,
        state: RuntimeState,
        memory_store: MemoryStore,
        relation_store: RelationStore,
    ) -> None:
        with self._lock:
            with self._connection:
                self._insert_transition(outcome.transition)
                self._persist_runtime_tables(
                    state=state,
                    memory_store=memory_store,
                    relation_store=relation_store,
                )

    def turn_count(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) AS count FROM turn_events WHERE speaker = 'user'"
            ).fetchone()
        return int(row["count"]) if row is not None else 0

    def phase_transition_count(self) -> int:
        with self._lock:
            row = self._connection.execute("SELECT COUNT(*) AS count FROM phase_events").fetchone()
        return int(row["count"]) if row is not None else 0

    def _insert_turn(self, turn: Turn) -> None:
        self._connection.execute(
            "INSERT INTO turn_events(turn_id, relation_id, session_id, speaker, text, created_at) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            (
                turn.turn_id,
                turn.relation_id,
                turn.session_id,
                turn.speaker.value,
                turn.text,
                turn.created_at,
            ),
        )

    def _insert_transition(self, transition: PhaseTransition) -> None:
        self._connection.execute(
            "INSERT INTO phase_events(transition_id, from_phase, to_phase, reason, created_at) "
            "VALUES(?, ?, ?, ?, ?)",
            (
                transition.transition_id,
                transition.from_phase.value,
                transition.to_phase.value,
                transition.reason,
                transition.created_at,
            ),
        )

    def _persist_runtime_tables(
        self,
        state: RuntimeState,
        memory_store: MemoryStore,
        relation_store: RelationStore,
    ) -> None:
        for fid in memory_store._dirty_fragments:
            fragment = memory_store.fragments[fid]
            self._connection.execute(
                "INSERT OR REPLACE INTO fragments(fragment_id, relation_id, turn_id, surface, tags, vividness, salience, "
                "unresolvedness, thread_ids, knot_ids, created_at, last_touched_at, activation_count) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    fragment.fragment_id,
                    fragment.relation_id,
                    fragment.turn_id,
                    fragment.surface,
                    _dump_json(fragment.tags),
                    fragment.vividness,
                    fragment.salience,
                    fragment.unresolvedness,
                    _dump_json(fragment.thread_ids),
                    _dump_json(fragment.knot_ids),
                    fragment.created_at,
                    fragment.last_touched_at,
                    fragment.activation_count,
                ),
            )

        for tid in memory_store._dirty_traces:
            trace = memory_store.traces[tid]
            self._connection.execute(
                "INSERT OR REPLACE INTO traces(trace_id, relation_id, fragment_id, channel, intensity, carry, created_at, last_touched_at) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trace.trace_id,
                    trace.relation_id,
                    trace.fragment_id,
                    trace.channel.value,
                    trace.intensity,
                    trace.carry,
                    trace.created_at,
                    trace.last_touched_at,
                ),
            )

        for eid in memory_store._dirty_associations:
            association = memory_store.associations[eid]
            self._connection.execute(
                "INSERT OR REPLACE INTO associations(edge_id, src_fragment_id, dst_fragment_id, kind, weight, evidence, created_at, last_touched_at) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    association.edge_id,
                    association.src_fragment_id,
                    association.dst_fragment_id,
                    association.kind.value,
                    association.weight,
                    _dump_json(association.evidence),
                    association.created_at,
                    association.last_touched_at,
                ),
            )

        for thid in memory_store._dirty_threads:
            thread = memory_store.threads[thid]
            self._connection.execute(
                "INSERT OR REPLACE INTO threads(thread_id, relation_id, fragment_ids, dominant_channels, tension, coherence, created_at, last_rewoven_at) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    thread.thread_id,
                    thread.relation_id,
                    _dump_json(thread.fragment_ids),
                    _dump_json(tuple(channel.value for channel in thread.dominant_channels)),
                    thread.tension,
                    thread.coherence,
                    thread.created_at,
                    thread.last_rewoven_at,
                ),
            )

        for kid in memory_store._dirty_knots:
            knot = memory_store.knots[kid]
            self._connection.execute(
                "INSERT OR REPLACE INTO knots(knot_id, relation_id, fragment_ids, dominant_channels, intensity, resolved, created_at, last_rewoven_at) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    knot.knot_id,
                    knot.relation_id,
                    _dump_json(knot.fragment_ids),
                    _dump_json(tuple(channel.value for channel in knot.dominant_channels)),
                    knot.intensity,
                    1 if knot.resolved else 0,
                    knot.created_at,
                    knot.last_rewoven_at,
                ),
            )

        for rel_id in relation_store._dirty_moment_relations:
            for moment in relation_store.moments.get(rel_id, ()):
                self._connection.execute(
                    "INSERT OR REPLACE INTO relation_moments(moment_id, relation_id, user_turn_id, aurora_turn_id, user_channels, aurora_move, boundary_event, repair_event, summary, created_at) "
                    "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        moment.moment_id,
                        moment.relation_id,
                        moment.user_turn_id,
                        moment.aurora_turn_id,
                        _dump_json(tuple(channel.value for channel in moment.user_channels)),
                        moment.aurora_move,
                        1 if moment.boundary_event else 0,
                        1 if moment.repair_event else 0,
                        moment.summary,
                        moment.created_at,
                    ),
                )

        for rel_id in relation_store._dirty_formations:
            formation = relation_store.formations[rel_id]
            self._connection.execute(
                "INSERT OR REPLACE INTO relation_formations(relation_id, thread_ids, knot_ids, boundary_events, repair_events, resonance_events, last_contact_at) "
                "VALUES(?, ?, ?, ?, ?, ?, ?)",
                (
                    formation.relation_id,
                    _dump_json(tuple(sorted(formation.thread_ids))),
                    _dump_json(tuple(sorted(formation.knot_ids))),
                    formation.boundary_events,
                    formation.repair_events,
                    formation.resonance_events,
                    formation.last_contact_at,
                ),
            )

        for fid in memory_store._deleted_fragments:
            self._connection.execute("DELETE FROM fragments WHERE fragment_id = ?", (fid,))
        for tid in memory_store._deleted_traces:
            self._connection.execute("DELETE FROM traces WHERE trace_id = ?", (tid,))
        for eid in memory_store._deleted_associations:
            self._connection.execute("DELETE FROM associations WHERE edge_id = ?", (eid,))
        for thid in memory_store._deleted_threads:
            self._connection.execute("DELETE FROM threads WHERE thread_id = ?", (thid,))
        for kid in memory_store._deleted_knots:
            self._connection.execute("DELETE FROM knots WHERE knot_id = ?", (kid,))

        memory_store.clear_dirty()
        relation_store.clear_dirty()

        orientation = state.orientation
        metabolic = state.metabolic

        self._connection.execute(
            "INSERT INTO orientation_state(id, self_evidence, world_evidence, relation_evidence, anchor_thread_ids, active_knot_ids, last_updated_at) "
            "VALUES(1, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET self_evidence = excluded.self_evidence, "
            "world_evidence = excluded.world_evidence, relation_evidence = excluded.relation_evidence, "
            "anchor_thread_ids = excluded.anchor_thread_ids, active_knot_ids = excluded.active_knot_ids, "
            "last_updated_at = excluded.last_updated_at",
            (
                _dump_json(orientation.self_evidence),
                _dump_json(orientation.world_evidence),
                _dump_json(orientation.relation_evidence),
                _dump_json(orientation.anchor_thread_ids),
                _dump_json(orientation.active_knot_ids),
                orientation.last_updated_at,
            ),
        )

        self._connection.execute(
            "INSERT INTO metabolic_state(id, phase, sleep_need, active_relation_ids, active_knot_ids, pending_sleep_relation_ids, last_transition_at) "
            "VALUES(1, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET phase = excluded.phase, sleep_need = excluded.sleep_need, "
            "active_relation_ids = excluded.active_relation_ids, active_knot_ids = excluded.active_knot_ids, "
            "pending_sleep_relation_ids = excluded.pending_sleep_relation_ids, "
            "last_transition_at = excluded.last_transition_at",
            (
                metabolic.phase.value,
                metabolic.sleep_need,
                _dump_json(metabolic.active_relation_ids),
                _dump_json(metabolic.active_knot_ids),
                _dump_json(metabolic.pending_sleep_relation_ids),
                metabolic.last_transition_at,
            ),
        )


def _dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _load_tuple_str(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    value = cast(list[Any], json.loads(str(raw)))
    return tuple(str(item) for item in value)


def _load_evidence_map(raw: Any) -> dict[str, tuple[str, ...]]:
    value = cast(dict[str, Any], json.loads(str(raw)))
    return {
        str(key): tuple(str(entry) for entry in item) if isinstance(item, list) else ()
        for key, item in value.items()
    }
