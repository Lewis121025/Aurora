"""Aurora SQLite persistence."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
import json
import os
import sqlite3
from typing import Any, cast

from aurora.runtime.contracts import (
    AtomKind,
    AtomStatus,
    MemoryAtom,
    SemanticScope,
    atom_content_from_dict,
    atom_content_to_dict,
)


def _json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def _json_loads(raw: str) -> Any:
    return json.loads(raw) if raw else {}


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
    """Unified SQLite store for memory atoms."""

    def __init__(self, db_path: str) -> None:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._tx_depth = 0
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_atoms (
                atom_id TEXT PRIMARY KEY,
                subject_id TEXT NOT NULL,
                atom_kind TEXT NOT NULL,
                content TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                salience REAL NOT NULL,
                accessibility REAL NOT NULL,
                scope TEXT,
                source_atom_ids TEXT NOT NULL,
                supersedes_atom_id TEXT,
                inhibits_atom_ids TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_atoms_subject_kind
            ON memory_atoms(subject_id, atom_kind, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_atoms_subject_status
            ON memory_atoms(subject_id, status, updated_at DESC);
            """
        )
        self.conn.execute(
            """
            INSERT INTO metadata(key, value) VALUES ('schema_version', 'atom-vnext')
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
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

    def subject_count(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(DISTINCT subject_id) AS count FROM memory_atoms"
        ).fetchone()
        return int(row["count"])

    def add_atom(self, atom: MemoryAtom) -> MemoryAtom:
        self.conn.execute(
            """
            INSERT INTO memory_atoms (
                atom_id, subject_id, atom_kind, content, status, confidence, salience,
                accessibility, scope, source_atom_ids, supersedes_atom_id,
                inhibits_atom_ids, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                atom.atom_id,
                atom.subject_id,
                atom.atom_kind,
                _json_dumps(atom_content_to_dict(atom.content)),
                atom.status,
                atom.confidence,
                atom.salience,
                atom.accessibility,
                atom.scope,
                _json_dumps(atom.source_atom_ids),
                atom.supersedes_atom_id,
                _json_dumps(atom.inhibits_atom_ids),
                atom.created_at,
                atom.updated_at,
            ),
        )
        self._commit()
        return atom

    def update_atom(self, atom: MemoryAtom) -> MemoryAtom:
        self.conn.execute(
            """
            UPDATE memory_atoms
            SET atom_kind = ?, content = ?, status = ?, confidence = ?, salience = ?,
                accessibility = ?, scope = ?, source_atom_ids = ?, supersedes_atom_id = ?,
                inhibits_atom_ids = ?, created_at = ?, updated_at = ?
            WHERE atom_id = ?
            """,
            (
                atom.atom_kind,
                _json_dumps(atom_content_to_dict(atom.content)),
                atom.status,
                atom.confidence,
                atom.salience,
                atom.accessibility,
                atom.scope,
                _json_dumps(atom.source_atom_ids),
                atom.supersedes_atom_id,
                _json_dumps(atom.inhibits_atom_ids),
                atom.created_at,
                atom.updated_at,
                atom.atom_id,
            ),
        )
        self._commit()
        return atom

    def get_atom(self, atom_id: str) -> MemoryAtom | None:
        row = self.conn.execute(
            "SELECT * FROM memory_atoms WHERE atom_id = ?",
            (atom_id,),
        ).fetchone()
        return None if row is None else self._atom_from_row(row)

    def list_atoms(
        self,
        subject_id: str,
        *,
        atom_kinds: Iterable[AtomKind] | None = None,
        include_inhibited: bool = True,
    ) -> tuple[MemoryAtom, ...]:
        sql = "SELECT * FROM memory_atoms WHERE subject_id = ?"
        params: list[object] = [subject_id]
        if atom_kinds:
            values = tuple(atom_kinds)
            placeholders = ", ".join("?" for _ in values)
            sql += f" AND atom_kind IN ({placeholders})"
            params.extend(values)
        if not include_inhibited:
            sql += " AND status != 'inhibited' AND accessibility > 0.0"
        sql += " ORDER BY updated_at ASC, created_at ASC"
        rows = self.conn.execute(sql, params).fetchall()
        return tuple(self._atom_from_row(row) for row in rows)

    def _atom_from_row(self, row: sqlite3.Row) -> MemoryAtom:
        scope = row["scope"]
        return MemoryAtom(
            atom_id=str(row["atom_id"]),
            subject_id=str(row["subject_id"]),
            atom_kind=cast(AtomKind, str(row["atom_kind"])),
            content=atom_content_from_dict(cast(AtomKind, str(row["atom_kind"])), _json_dict(str(row["content"]))),
            status=cast(AtomStatus, str(row["status"])),
            confidence=float(row["confidence"]),
            salience=float(row["salience"]),
            accessibility=float(row["accessibility"]),
            scope=None if scope is None else cast(SemanticScope, str(scope)),
            source_atom_ids=tuple(_json_list(str(row["source_atom_ids"]))),
            supersedes_atom_id=None if row["supersedes_atom_id"] is None else str(row["supersedes_atom_id"]),
            inhibits_atom_ids=tuple(_json_list(str(row["inhibits_atom_ids"]))),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    def _commit(self) -> None:
        if self._tx_depth == 0:
            self.conn.commit()
