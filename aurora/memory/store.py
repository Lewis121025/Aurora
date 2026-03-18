"""Aurora v3 SQLite persistence."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
import json
import os
import sqlite3
from typing import Any, cast
from uuid import uuid4

from aurora.runtime.contracts import AtomStatus, AtomType, EventKind, EventRecord, MemoryAtom


def _json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def _json_loads(raw: str) -> Any:
    return json.loads(raw) if raw else []


def _json_list(raw: str) -> list[str]:
    data = _json_loads(raw)
    if not isinstance(data, list):
        return []
    return [str(item) for item in data]


def _json_dict(raw: str) -> dict[str, Any]:
    data = _json_loads(raw)
    if not isinstance(data, dict):
        return {}
    return {str(key): value for key, value in data.items()}


class SQLiteMemoryStore:
    """Unified SQLite store for evidence events and memory atoms."""

    def __init__(self, db_path: str) -> None:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._tx_depth = 0
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                relation_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_relation_created
            ON events(relation_id, created_at);

            CREATE TABLE IF NOT EXISTS memory_atoms (
                atom_id TEXT PRIMARY KEY,
                relation_id TEXT NOT NULL,
                atom_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                salience REAL NOT NULL,
                visibility REAL NOT NULL,
                evidence_event_ids TEXT NOT NULL,
                affects_atom_ids TEXT NOT NULL,
                supersedes_atom_id TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_atoms_relation_type
            ON memory_atoms(relation_id, atom_type, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_atoms_relation_visibility
            ON memory_atoms(relation_id, visibility, updated_at DESC);
            """
        )
        self._commit()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        if self._tx_depth == 0:
            self.conn.execute("BEGIN")
        self._tx_depth += 1
        try:
            yield
        except Exception:
            self._tx_depth -= 1
            if self._tx_depth == 0:
                self.conn.rollback()
            raise
        else:
            self._tx_depth -= 1
            if self._tx_depth == 0:
                self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def relation_count(self) -> int:
        row = self.conn.execute(
            """
            SELECT COUNT(*) AS count FROM (
                SELECT relation_id FROM events
                UNION
                SELECT relation_id FROM memory_atoms
            )
            """
        ).fetchone()
        return int(row["count"])

    def append_event(
        self,
        *,
        relation_id: str,
        kind: EventKind,
        role: str,
        text: str,
        created_at: float,
        payload: dict[str, object] | None = None,
    ) -> EventRecord:
        event = EventRecord(
            event_id=f"evt_{uuid4().hex[:12]}",
            relation_id=relation_id,
            kind=kind,
            role=role,
            text=text,
            created_at=created_at,
            payload=payload or {},
        )
        self.conn.execute(
            """
            INSERT INTO events (
                event_id, relation_id, kind, role, text, payload, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.relation_id,
                event.kind,
                event.role,
                event.text,
                _json_dumps(event.payload),
                event.created_at,
            ),
        )
        self._commit()
        return event

    def recent_turns(self, relation_id: str, limit: int = 6) -> tuple[EventRecord, ...]:
        rows = self.conn.execute(
            """
            SELECT * FROM events
            WHERE relation_id = ? AND kind IN ('user_turn', 'assistant_turn')
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (relation_id, limit),
        ).fetchall()
        return tuple(reversed([self._event_from_row(row) for row in rows]))

    def add_atom(self, atom: MemoryAtom) -> MemoryAtom:
        self.conn.execute(
            """
            INSERT INTO memory_atoms (
                atom_id, relation_id, atom_type, payload, status, confidence, salience,
                visibility, evidence_event_ids, affects_atom_ids, supersedes_atom_id,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                atom.atom_id,
                atom.relation_id,
                atom.atom_type,
                _json_dumps(atom.payload),
                atom.status,
                atom.confidence,
                atom.salience,
                atom.visibility,
                _json_dumps(atom.evidence_event_ids),
                _json_dumps(atom.affects_atom_ids),
                atom.supersedes_atom_id,
                atom.created_at,
                atom.updated_at,
            ),
        )
        self._commit()
        return atom

    def list_atoms(
        self,
        relation_id: str,
        *,
        atom_types: Iterable[AtomType] | None = None,
        include_hidden: bool = True,
    ) -> tuple[MemoryAtom, ...]:
        sql = "SELECT * FROM memory_atoms WHERE relation_id = ?"
        params: list[object] = [relation_id]
        if atom_types:
            values = tuple(atom_types)
            placeholders = ", ".join("?" for _ in values)
            sql += f" AND atom_type IN ({placeholders})"
            params.extend(values)
        if not include_hidden:
            sql += " AND status != 'hidden' AND visibility > 0.0"
        sql += " ORDER BY updated_at DESC, created_at DESC"
        rows = self.conn.execute(sql, params).fetchall()
        return tuple(self._atom_from_row(row) for row in rows)

    def get_atom(self, atom_id: str) -> MemoryAtom | None:
        row = self.conn.execute(
            "SELECT * FROM memory_atoms WHERE atom_id = ?",
            (atom_id,),
        ).fetchone()
        return None if row is None else self._atom_from_row(row)

    def _event_from_row(self, row: sqlite3.Row) -> EventRecord:
        return EventRecord(
            event_id=str(row["event_id"]),
            relation_id=str(row["relation_id"]),
            kind=cast(EventKind, str(row["kind"])),
            role=str(row["role"]),
            text=str(row["text"]),
            created_at=float(row["created_at"]),
            payload=_json_dict(str(row["payload"])),
        )

    def _atom_from_row(self, row: sqlite3.Row) -> MemoryAtom:
        return MemoryAtom(
            atom_id=str(row["atom_id"]),
            relation_id=str(row["relation_id"]),
            atom_type=cast(AtomType, str(row["atom_type"])),
            payload=_json_dict(str(row["payload"])),
            status=cast(AtomStatus, str(row["status"])),
            confidence=float(row["confidence"]),
            salience=float(row["salience"]),
            visibility=float(row["visibility"]),
            evidence_event_ids=tuple(_json_list(str(row["evidence_event_ids"]))),
            affects_atom_ids=tuple(_json_list(str(row["affects_atom_ids"]))),
            supersedes_atom_id=None if row["supersedes_atom_id"] is None else str(row["supersedes_atom_id"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    def _commit(self) -> None:
        if self._tx_depth == 0:
            self.conn.commit()
