from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import cast

from aurora.memory.store import MemoryStore
from aurora.persistence.migrations import apply_migrations
from aurora.phases.phase_types import Phase
from aurora.relation.store import RelationStore
from aurora.runtime.models import (
    AssociationDelta,
    AwakeOutcome,
    ExistentialSnapshot,
    Fragment,
    InteractionTurn,
    PhaseOutcome,
    PhaseTransition,
    RelationMoment,
    RuntimeState,
    Tone,
    Trace,
)


def _phase_from_str(value: str) -> Phase:
    return Phase(value)


def _tone_from_str(value: str) -> Tone:
    return cast(Tone, value)


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
        self, initial: ExistentialSnapshot
    ) -> tuple[MemoryStore, RelationStore, RuntimeState]:
        memory_store = MemoryStore()
        relation_store = RelationStore()

        with self._lock:
            for row in self._connection.execute(
                "SELECT fragment_id, turn_id, text, created_at, salience, narrative_weight "
                "FROM fragments ORDER BY created_at"
            ):
                fragment = Fragment(
                    fragment_id=str(row["fragment_id"]),
                    turn_id=str(row["turn_id"]),
                    text=str(row["text"]),
                    created_at=float(row["created_at"]),
                )
                memory_store.fragments.append(fragment)
                memory_store.fragment_salience[fragment.fragment_id] = float(row["salience"])
                memory_store.fragment_narrative_weight[fragment.fragment_id] = float(
                    row["narrative_weight"]
                )

            for row in self._connection.execute(
                "SELECT trace_id, turn_id, mode, intensity, created_at FROM traces ORDER BY created_at"
            ):
                memory_store.traces.append(
                    Trace(
                        trace_id=str(row["trace_id"]),
                        turn_id=str(row["turn_id"]),
                        mode=str(row["mode"]),
                        intensity=float(row["intensity"]),
                        created_at=float(row["created_at"]),
                    )
                )

            for row in self._connection.execute(
                "SELECT association_id, source_fragment_id, target_fragment_id, weight, created_at "
                "FROM associations ORDER BY created_at"
            ):
                memory_store.associations.append(
                    AssociationDelta(
                        association_id=str(row["association_id"]),
                        source_fragment_id=str(row["source_fragment_id"]),
                        target_fragment_id=str(row["target_fragment_id"]),
                        weight=float(row["weight"]),
                        created_at=float(row["created_at"]),
                    )
                )

            for row in self._connection.execute(
                "SELECT moment_id, session_id, turn_id, tone, summary, created_at "
                "FROM relation_moments ORDER BY created_at"
            ):
                relation_store.moments.append(
                    RelationMoment(
                        moment_id=str(row["moment_id"]),
                        session_id=str(row["session_id"]),
                        turn_id=str(row["turn_id"]),
                        tone=_tone_from_str(str(row["tone"])),
                        summary=str(row["summary"]),
                        created_at=float(row["created_at"]),
                    )
                )

            runtime_meta = {
                str(row["meta_key"]): str(row["meta_value"])
                for row in self._connection.execute("SELECT meta_key, meta_value FROM runtime_meta")
            }
        memory_store.sleep_cycles = int(runtime_meta.get("sleep_cycles", "0"))
        memory_store.last_reweave_delta = float(runtime_meta.get("last_reweave_delta", "0.0"))

        with self._lock:
            latest_snapshot_row = self._connection.execute(
                "SELECT phase, self_view, world_view, openness, updated_at "
                "FROM snapshots ORDER BY snapshot_id DESC LIMIT 1"
            ).fetchone()
        if latest_snapshot_row is None:
            snapshot = initial
        else:
            snapshot = ExistentialSnapshot(
                phase=_phase_from_str(str(latest_snapshot_row["phase"])),
                self_view=float(latest_snapshot_row["self_view"]),
                world_view=float(latest_snapshot_row["world_view"]),
                openness=float(latest_snapshot_row["openness"]),
                updated_at=float(latest_snapshot_row["updated_at"]),
            )

        with self._lock:
            transitions = [
                PhaseTransition(
                    transition_id=str(row["transition_id"]),
                    from_phase=_phase_from_str(str(row["from_phase"])),
                    to_phase=_phase_from_str(str(row["to_phase"])),
                    reason=str(row["reason"]),
                    created_at=float(row["created_at"]),
                )
                for row in self._connection.execute(
                    "SELECT transition_id, from_phase, to_phase, reason, created_at "
                    "FROM phase_transitions ORDER BY created_at"
                )
            ]

        return (
            memory_store,
            relation_store,
            RuntimeState(snapshot=snapshot, transitions=transitions),
        )

    def persist_awake(
        self, turn: InteractionTurn, outcome: AwakeOutcome, memory_store: MemoryStore
    ) -> None:
        with self._lock:
            with self._connection:
                self._connection.execute(
                    "INSERT OR REPLACE INTO turns(turn_id, session_id, speaker, text, created_at) "
                    "VALUES(?, ?, ?, ?, ?)",
                    (turn.turn_id, turn.session_id, turn.speaker, turn.text, turn.created_at),
                )
                self._connection.execute(
                    "INSERT OR REPLACE INTO fragments(fragment_id, turn_id, text, created_at, salience, narrative_weight) "
                    "VALUES(?, ?, ?, ?, ?, ?)",
                    (
                        outcome.fragment.fragment_id,
                        outcome.fragment.turn_id,
                        outcome.fragment.text,
                        outcome.fragment.created_at,
                        memory_store.fragment_salience.get(outcome.fragment.fragment_id, 0.4),
                        memory_store.fragment_narrative_weight.get(
                            outcome.fragment.fragment_id, 0.4
                        ),
                    ),
                )

                for trace in outcome.traces:
                    self._connection.execute(
                        "INSERT OR REPLACE INTO traces(trace_id, turn_id, mode, intensity, created_at) "
                        "VALUES(?, ?, ?, ?, ?)",
                        (
                            trace.trace_id,
                            trace.turn_id,
                            trace.mode,
                            trace.intensity,
                            trace.created_at,
                        ),
                    )

                for association in outcome.associations:
                    self._connection.execute(
                        "INSERT OR REPLACE INTO associations(association_id, source_fragment_id, "
                        "target_fragment_id, weight, created_at) VALUES(?, ?, ?, ?, ?)",
                        (
                            association.association_id,
                            association.source_fragment_id,
                            association.target_fragment_id,
                            association.weight,
                            association.created_at,
                        ),
                    )

                moment = outcome.relation_moment
                self._connection.execute(
                    "INSERT OR REPLACE INTO relation_moments(moment_id, session_id, turn_id, tone, summary, "
                    "created_at) VALUES(?, ?, ?, ?, ?, ?)",
                    (
                        moment.moment_id,
                        moment.session_id,
                        moment.turn_id,
                        moment.tone,
                        moment.summary,
                        moment.created_at,
                    ),
                )

                if outcome.transition is not None:
                    self._persist_transition(outcome.transition)

                self._persist_snapshot(outcome.snapshot)
                self._persist_memory_store(memory_store)

    def persist_phase(self, outcome: PhaseOutcome, memory_store: MemoryStore) -> None:
        with self._lock:
            with self._connection:
                self._persist_transition(outcome.transition)
                self._persist_snapshot(outcome.snapshot)
                self._persist_memory_store(memory_store)

    def turn_count(self) -> int:
        with self._lock:
            row = self._connection.execute("SELECT COUNT(*) AS count FROM turns").fetchone()
        if row is None:
            return 0
        return int(row["count"])

    def phase_transition_count(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) AS count FROM phase_transitions"
            ).fetchone()
        if row is None:
            return 0
        return int(row["count"])

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

    def _persist_snapshot(self, snapshot: ExistentialSnapshot) -> None:
        self._connection.execute(
            "INSERT INTO snapshots(phase, self_view, world_view, openness, updated_at) VALUES(?, ?, ?, ?, ?)",
            (
                snapshot.phase.value,
                snapshot.self_view,
                snapshot.world_view,
                snapshot.openness,
                snapshot.updated_at,
            ),
        )

    def _persist_memory_metrics(self, memory_store: MemoryStore) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO runtime_meta(meta_key, meta_value) VALUES('sleep_cycles', ?)",
            (str(memory_store.sleep_cycles),),
        )
        self._connection.execute(
            "INSERT OR REPLACE INTO runtime_meta(meta_key, meta_value) VALUES('last_reweave_delta', ?)",
            (str(memory_store.last_reweave_delta),),
        )

    def _persist_all_fragments(self, memory_store: MemoryStore) -> None:
        for fragment in memory_store.fragments:
            self._connection.execute(
                "UPDATE fragments SET salience = ?, narrative_weight = ? WHERE fragment_id = ?",
                (
                    memory_store.fragment_salience.get(fragment.fragment_id, 0.4),
                    memory_store.fragment_narrative_weight.get(fragment.fragment_id, 0.4),
                    fragment.fragment_id,
                ),
            )

    def _persist_all_associations(self, memory_store: MemoryStore) -> None:
        for association in memory_store.associations:
            self._connection.execute(
                "UPDATE associations SET weight = ? WHERE association_id = ?",
                (association.weight, association.association_id),
            )

    def _persist_memory_store(self, memory_store: MemoryStore) -> None:
        self._persist_all_fragments(memory_store)
        self._persist_all_associations(memory_store)
        self._persist_memory_metrics(memory_store)
