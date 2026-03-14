from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from aurora.memory.store import MemoryStore
from aurora.persistence.migrations import apply_migrations
from aurora.relation.store import RelationStore
from aurora.runtime.models import (
    Association,
    AssocKind,
    AwakeOutcome,
    Fragment,
    Knot,
    MetabolicState,
    Orientation,
    Phase,
    PhaseOutcome,
    PhaseTransition,
    RelationFormation,
    RelationMoment,
    RelationMove,
    RuntimeState,
    Speaker,
    Thread,
    Trace,
    TraceChannel,
    Turn,
)


def _encode(values: tuple[str, ...]) -> str:
    return json.dumps(list(values), ensure_ascii=True)


def _decode(raw: str) -> tuple[str, ...]:
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        return ()
    return tuple(str(item) for item in parsed)


class SQLitePersistence:
    def __init__(self, data_dir: str | None = None, db_name: str = "aurora.sqlite3") -> None:
        base_dir = Path(data_dir) if data_dir is not None else Path(".aurora")
        base_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = base_dir / db_name
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        apply_migrations(self._connection)

    def load_runtime(
        self, initial: MetabolicState
    ) -> tuple[MemoryStore, RelationStore, RuntimeState]:
        memory_store = MemoryStore()
        relation_store = RelationStore()

        with self._lock:
            for row in self._connection.execute(
                "SELECT fragment_id, turn_id, relation_id, surface, touch_channels, salience, vividness, "
                "unresolvedness, activation, created_at, last_touched_at FROM fragments"
            ):
                memory_store.fragments[str(row["fragment_id"])] = Fragment(
                    fragment_id=str(row["fragment_id"]),
                    turn_id=str(row["turn_id"]),
                    relation_id=str(row["relation_id"]),
                    surface=str(row["surface"]),
                    touch_channels=tuple(
                        TraceChannel(channel) for channel in _decode(str(row["touch_channels"]))
                    ),
                    salience=float(row["salience"]),
                    vividness=float(row["vividness"]),
                    unresolvedness=float(row["unresolvedness"]),
                    activation=float(row["activation"]),
                    created_at=float(row["created_at"]),
                    last_touched_at=float(row["last_touched_at"]),
                )

            for row in self._connection.execute(
                "SELECT trace_id, fragment_id, relation_id, channel, intensity, persistence, created_at, "
                "last_touched_at FROM traces"
            ):
                trace = Trace(
                    trace_id=str(row["trace_id"]),
                    fragment_id=str(row["fragment_id"]),
                    relation_id=str(row["relation_id"]),
                    channel=TraceChannel(str(row["channel"])),
                    intensity=float(row["intensity"]),
                    persistence=float(row["persistence"]),
                    created_at=float(row["created_at"]),
                    last_touched_at=float(row["last_touched_at"]),
                )
                memory_store.traces[trace.trace_id] = trace

            for row in self._connection.execute(
                "SELECT edge_id, src_fragment_id, dst_fragment_id, kind, weight, evidence_count, created_at, "
                "last_touched_at FROM associations"
            ):
                memory_store.associations[str(row["edge_id"])] = Association(
                    edge_id=str(row["edge_id"]),
                    src_fragment_id=str(row["src_fragment_id"]),
                    dst_fragment_id=str(row["dst_fragment_id"]),
                    kind=AssocKind(str(row["kind"])),
                    weight=float(row["weight"]),
                    evidence_count=int(row["evidence_count"]),
                    created_at=float(row["created_at"]),
                    last_touched_at=float(row["last_touched_at"]),
                )

            for row in self._connection.execute(
                "SELECT thread_id, relation_id, fragment_ids, motif_channels, coherence, tension, synopsis, "
                "created_at, last_rewoven_at FROM threads"
            ):
                thread = Thread(
                    thread_id=str(row["thread_id"]),
                    relation_id=str(row["relation_id"]),
                    fragment_ids=_decode(str(row["fragment_ids"])),
                    motif_channels=tuple(
                        TraceChannel(channel) for channel in _decode(str(row["motif_channels"]))
                    ),
                    coherence=float(row["coherence"]),
                    tension=float(row["tension"]),
                    synopsis=str(row["synopsis"]),
                    created_at=float(row["created_at"]),
                    last_rewoven_at=float(row["last_rewoven_at"]),
                )
                memory_store.threads[thread.thread_id] = thread

            for row in self._connection.execute(
                "SELECT knot_id, relation_id, fragment_ids, channel, density, heat, created_at, "
                "last_touched_at FROM knots"
            ):
                knot = Knot(
                    knot_id=str(row["knot_id"]),
                    relation_id=str(row["relation_id"]),
                    fragment_ids=_decode(str(row["fragment_ids"])),
                    channel=TraceChannel(str(row["channel"])),
                    density=float(row["density"]),
                    heat=float(row["heat"]),
                    created_at=float(row["created_at"]),
                    last_touched_at=float(row["last_touched_at"]),
                )
                memory_store.knots[knot.knot_id] = knot

            for row in self._connection.execute(
                "SELECT moment_id, relation_id, session_id, user_turn_id, aurora_turn_id, user_channels, "
                "user_move, aurora_move, boundary_signal, resonance_score, note, created_at "
                "FROM relation_moments"
            ):
                moment = RelationMoment(
                    moment_id=str(row["moment_id"]),
                    relation_id=str(row["relation_id"]),
                    session_id=str(row["session_id"]),
                    user_turn_id=str(row["user_turn_id"]),
                    aurora_turn_id=str(row["aurora_turn_id"]),
                    user_channels=tuple(
                        TraceChannel(channel) for channel in _decode(str(row["user_channels"]))
                    ),
                    user_move=RelationMove(str(row["user_move"])),
                    aurora_move=RelationMove(str(row["aurora_move"])),
                    boundary_signal=float(row["boundary_signal"]),
                    resonance_score=float(row["resonance_score"]),
                    note=str(row["note"]),
                    created_at=float(row["created_at"]),
                )
                relation_store.moments[moment.moment_id] = moment

            for row in self._connection.execute(
                "SELECT relation_id, trust, familiarity, reciprocity, boundary_tension, repairability, "
                "active_thread_ids, active_knot_ids, last_contact_at FROM relation_formations"
            ):
                formation = RelationFormation(
                    relation_id=str(row["relation_id"]),
                    trust=float(row["trust"]),
                    familiarity=float(row["familiarity"]),
                    reciprocity=float(row["reciprocity"]),
                    boundary_tension=float(row["boundary_tension"]),
                    repairability=float(row["repairability"]),
                    active_thread_ids=_decode(str(row["active_thread_ids"])),
                    active_knot_ids=_decode(str(row["active_knot_ids"])),
                    last_contact_at=float(row["last_contact_at"]),
                )
                relation_store.formations[formation.relation_id] = formation

            for row in self._connection.execute(
                "SELECT relation_id, self_orientation, world_orientation, relation_orientation, "
                "narrative_tilt, updated_at FROM orientations"
            ):
                orientation = Orientation(
                    relation_id=str(row["relation_id"]),
                    self_orientation=float(row["self_orientation"]),
                    world_orientation=float(row["world_orientation"]),
                    relation_orientation=float(row["relation_orientation"]),
                    narrative_tilt=float(row["narrative_tilt"]),
                    updated_at=float(row["updated_at"]),
                )
                relation_store.orientations[orientation.relation_id] = orientation

            transition_rows = self._connection.execute(
                "SELECT transition_id, from_phase, to_phase, reason, created_at FROM phase_transitions "
                "ORDER BY created_at"
            )
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

            state_row = self._connection.execute(
                "SELECT phase, sleep_need, current_relation_id, active_thread_ids, active_knot_ids, "
                "last_transition_at FROM metabolic_state WHERE state_id = 1"
            ).fetchone()
            if state_row is None:
                metabolic = initial
            else:
                metabolic = MetabolicState(
                    phase=Phase(str(state_row["phase"])),
                    sleep_need=float(state_row["sleep_need"]),
                    current_relation_id=(
                        None
                        if state_row["current_relation_id"] is None
                        else str(state_row["current_relation_id"])
                    ),
                    active_thread_ids=_decode(str(state_row["active_thread_ids"])),
                    active_knot_ids=_decode(str(state_row["active_knot_ids"])),
                    last_transition_at=float(state_row["last_transition_at"]),
                )

            runtime_meta = {
                str(row["meta_key"]): str(row["meta_value"])
                for row in self._connection.execute("SELECT meta_key, meta_value FROM runtime_meta")
            }

        memory_store.sleep_cycles = int(runtime_meta.get("sleep_cycles", "0"))
        memory_store.last_reweave_delta = float(runtime_meta.get("last_reweave_delta", "0.0"))
        return (
            memory_store,
            relation_store,
            RuntimeState(metabolic=metabolic, transitions=transitions),
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
                self._persist_turn(outcome.user_turn)
                self._persist_turn(outcome.aurora_turn)
                if outcome.transition is not None:
                    self._persist_transition(outcome.transition)
                self._persist_all_state(
                    state=state, memory_store=memory_store, relation_store=relation_store
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
                self._persist_transition(outcome.transition)
                self._persist_all_state(
                    state=state, memory_store=memory_store, relation_store=relation_store
                )

    def turn_count(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) AS count FROM turns WHERE speaker = ?", (Speaker.USER.value,)
            ).fetchone()
        return int(row["count"]) if row is not None else 0

    def phase_transition_count(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) AS count FROM phase_transitions"
            ).fetchone()
        return int(row["count"]) if row is not None else 0

    def _persist_turn(self, turn: Turn) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO turns(turn_id, relation_id, session_id, speaker, text, created_at, "
            "reply_to_turn_id) VALUES(?, ?, ?, ?, ?, ?, ?)",
            (
                turn.turn_id,
                turn.relation_id,
                turn.session_id,
                turn.speaker.value,
                turn.text,
                turn.created_at,
                turn.reply_to_turn_id,
            ),
        )

    def _persist_transition(self, transition: PhaseTransition) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO phase_transitions(transition_id, from_phase, to_phase, reason, "
            "created_at) VALUES(?, ?, ?, ?, ?)",
            (
                transition.transition_id,
                transition.from_phase.value,
                transition.to_phase.value,
                transition.reason,
                transition.created_at,
            ),
        )

    def _persist_all_state(
        self,
        state: RuntimeState,
        memory_store: MemoryStore,
        relation_store: RelationStore,
    ) -> None:
        self._persist_metabolic(state.metabolic)
        self._persist_memory_store(memory_store)
        self._persist_relation_store(relation_store)
        self._connection.execute(
            "INSERT OR REPLACE INTO runtime_meta(meta_key, meta_value) VALUES('sleep_cycles', ?)",
            (str(memory_store.sleep_cycles),),
        )
        self._connection.execute(
            "INSERT OR REPLACE INTO runtime_meta(meta_key, meta_value) VALUES('last_reweave_delta', ?)",
            (str(memory_store.last_reweave_delta),),
        )

    def _persist_metabolic(self, metabolic: MetabolicState) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO metabolic_state(state_id, phase, sleep_need, current_relation_id, "
            "active_thread_ids, active_knot_ids, last_transition_at) VALUES(1, ?, ?, ?, ?, ?, ?)",
            (
                metabolic.phase.value,
                metabolic.sleep_need,
                metabolic.current_relation_id,
                _encode(metabolic.active_thread_ids),
                _encode(metabolic.active_knot_ids),
                metabolic.last_transition_at,
            ),
        )

    def _persist_memory_store(self, memory_store: MemoryStore) -> None:
        self._connection.execute("DELETE FROM fragments")
        self._connection.execute("DELETE FROM traces")
        self._connection.execute("DELETE FROM associations")
        self._connection.execute("DELETE FROM threads")
        self._connection.execute("DELETE FROM knots")

        for fragment in memory_store.fragments.values():
            self._connection.execute(
                "INSERT INTO fragments(fragment_id, turn_id, relation_id, surface, touch_channels, salience, "
                "vividness, unresolvedness, activation, created_at, last_touched_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    fragment.fragment_id,
                    fragment.turn_id,
                    fragment.relation_id,
                    fragment.surface,
                    _encode(tuple(channel.value for channel in fragment.touch_channels)),
                    fragment.salience,
                    fragment.vividness,
                    fragment.unresolvedness,
                    fragment.activation,
                    fragment.created_at,
                    fragment.last_touched_at,
                ),
            )

        for trace in memory_store.traces.values():
            self._connection.execute(
                "INSERT INTO traces(trace_id, fragment_id, relation_id, channel, intensity, persistence, "
                "created_at, last_touched_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trace.trace_id,
                    trace.fragment_id,
                    trace.relation_id,
                    trace.channel.value,
                    trace.intensity,
                    trace.persistence,
                    trace.created_at,
                    trace.last_touched_at,
                ),
            )

        for association in memory_store.associations.values():
            self._connection.execute(
                "INSERT INTO associations(edge_id, src_fragment_id, dst_fragment_id, kind, weight, evidence_count, "
                "created_at, last_touched_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    association.edge_id,
                    association.src_fragment_id,
                    association.dst_fragment_id,
                    association.kind.value,
                    association.weight,
                    association.evidence_count,
                    association.created_at,
                    association.last_touched_at,
                ),
            )

        for thread in memory_store.threads.values():
            self._connection.execute(
                "INSERT INTO threads(thread_id, relation_id, fragment_ids, motif_channels, coherence, tension, "
                "synopsis, created_at, last_rewoven_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    thread.thread_id,
                    thread.relation_id,
                    _encode(thread.fragment_ids),
                    _encode(tuple(channel.value for channel in thread.motif_channels)),
                    thread.coherence,
                    thread.tension,
                    thread.synopsis,
                    thread.created_at,
                    thread.last_rewoven_at,
                ),
            )

        for knot in memory_store.knots.values():
            self._connection.execute(
                "INSERT INTO knots(knot_id, relation_id, fragment_ids, channel, density, heat, created_at, "
                "last_touched_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    knot.knot_id,
                    knot.relation_id,
                    _encode(knot.fragment_ids),
                    knot.channel.value,
                    knot.density,
                    knot.heat,
                    knot.created_at,
                    knot.last_touched_at,
                ),
            )

    def _persist_relation_store(self, relation_store: RelationStore) -> None:
        self._connection.execute("DELETE FROM relation_moments")
        self._connection.execute("DELETE FROM relation_formations")
        self._connection.execute("DELETE FROM orientations")

        for moment in relation_store.moments.values():
            self._connection.execute(
                "INSERT INTO relation_moments(moment_id, relation_id, session_id, user_turn_id, aurora_turn_id, "
                "user_channels, user_move, aurora_move, boundary_signal, resonance_score, note, created_at) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    moment.moment_id,
                    moment.relation_id,
                    moment.session_id,
                    moment.user_turn_id,
                    moment.aurora_turn_id,
                    _encode(tuple(channel.value for channel in moment.user_channels)),
                    moment.user_move.value,
                    moment.aurora_move.value,
                    moment.boundary_signal,
                    moment.resonance_score,
                    moment.note,
                    moment.created_at,
                ),
            )

        for formation in relation_store.formations.values():
            self._connection.execute(
                "INSERT INTO relation_formations(relation_id, trust, familiarity, reciprocity, boundary_tension, "
                "repairability, active_thread_ids, active_knot_ids, last_contact_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    formation.relation_id,
                    formation.trust,
                    formation.familiarity,
                    formation.reciprocity,
                    formation.boundary_tension,
                    formation.repairability,
                    _encode(formation.active_thread_ids),
                    _encode(formation.active_knot_ids),
                    formation.last_contact_at,
                ),
            )

        for orientation in relation_store.orientations.values():
            self._connection.execute(
                "INSERT INTO orientations(relation_id, self_orientation, world_orientation, relation_orientation, "
                "narrative_tilt, updated_at) VALUES(?, ?, ?, ?, ?, ?)",
                (
                    orientation.relation_id,
                    orientation.self_orientation,
                    orientation.world_orientation,
                    orientation.relation_orientation,
                    orientation.narrative_tilt,
                    orientation.updated_at,
                ),
            )
