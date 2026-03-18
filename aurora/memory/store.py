"""Aurora SQLite persistence for the memory field kernel."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
import json
import os
import sqlite3
from typing import Any

from aurora.runtime.contracts import (
    AtomKind,
    MemoryAtom,
    MemoryEdge,
    atom_content_from_dict,
    atom_content_to_dict,
    atom_kind_from_value,
)

_SCHEMA_VERSION = "field-v4"


def _json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def _json_loads(raw: str) -> Any:
    return json.loads(raw) if raw else {}


def _json_list(raw: str) -> list[str]:
    data = _json_loads(raw)
    if not isinstance(data, list):
        raise ValueError("expected JSON array")
    return [str(item) for item in data]


def _json_dict(raw: str) -> dict[str, Any]:
    data = _json_loads(raw)
    if not isinstance(data, dict):
        raise ValueError("expected JSON object")
    return {str(key): value for key, value in data.items()}


class SQLiteMemoryStore:
    """SQLite store for immutable field atoms, edges, and activation cache."""

    def __init__(self, db_path: str) -> None:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._tx_depth = 0
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        row = self.conn.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
        current_version = "" if row is None else str(row["value"])
        if current_version != _SCHEMA_VERSION:
            self.conn.executescript(
                """
                DROP TABLE IF EXISTS memory_atoms;
                DROP TABLE IF EXISTS memory_edges;
                DROP TABLE IF EXISTS activation_cache;

                CREATE TABLE memory_atoms (
                    atom_id TEXT PRIMARY KEY CHECK(atom_id <> ''),
                    subject_id TEXT NOT NULL CHECK(subject_id <> ''),
                    atom_kind TEXT NOT NULL CHECK(atom_kind IN ('evidence', 'memory', 'episode', 'inhibition')),
                    content TEXT NOT NULL,
                    confidence REAL NOT NULL CHECK(confidence BETWEEN 0.0 AND 1.0),
                    salience REAL NOT NULL CHECK(salience BETWEEN 0.0 AND 1.0),
                    source_atom_ids TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    CHECK(atom_kind <> 'evidence' OR source_atom_ids = '[]'),
                    UNIQUE(subject_id, atom_id)
                );
                CREATE INDEX idx_atoms_subject_kind
                ON memory_atoms(subject_id, atom_kind, created_at DESC);
                CREATE INDEX idx_atoms_subject_created
                ON memory_atoms(subject_id, created_at DESC);

                CREATE TABLE memory_edges (
                    edge_id TEXT PRIMARY KEY CHECK(edge_id <> ''),
                    subject_id TEXT NOT NULL CHECK(subject_id <> ''),
                    source_atom_id TEXT NOT NULL,
                    target_atom_id TEXT NOT NULL,
                    influence REAL NOT NULL CHECK(influence BETWEEN -1.0 AND 1.0),
                    confidence REAL NOT NULL CHECK(confidence BETWEEN 0.0 AND 1.0),
                    created_at REAL NOT NULL,
                    CHECK(source_atom_id <> target_atom_id),
                    FOREIGN KEY(subject_id, source_atom_id) REFERENCES memory_atoms(subject_id, atom_id),
                    FOREIGN KEY(subject_id, target_atom_id) REFERENCES memory_atoms(subject_id, atom_id)
                );
                CREATE INDEX idx_edges_subject_source
                ON memory_edges(subject_id, source_atom_id, created_at DESC);
                CREATE INDEX idx_edges_subject_target
                ON memory_edges(subject_id, target_atom_id, created_at DESC);

                CREATE TRIGGER reject_memory_edges_touching_evidence
                BEFORE INSERT ON memory_edges
                FOR EACH ROW
                WHEN EXISTS (
                    SELECT 1
                    FROM memory_atoms
                    WHERE subject_id = NEW.subject_id
                      AND atom_kind = 'evidence'
                      AND atom_id IN (NEW.source_atom_id, NEW.target_atom_id)
                )
                BEGIN
                    SELECT RAISE(ABORT, 'memory_edges cannot touch evidence atoms');
                END;

                CREATE TABLE activation_cache (
                    subject_id TEXT NOT NULL CHECK(subject_id <> ''),
                    atom_id TEXT NOT NULL,
                    activation REAL NOT NULL CHECK(activation BETWEEN 0.0 AND 1.0),
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(subject_id, atom_id),
                    FOREIGN KEY(subject_id, atom_id) REFERENCES memory_atoms(subject_id, atom_id)
                );
                CREATE INDEX idx_activation_subject_value
                ON activation_cache(subject_id, activation DESC, updated_at DESC);
                """
            )
            self.conn.execute(
                """
                INSERT INTO metadata(key, value) VALUES ('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (_SCHEMA_VERSION,),
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
                atom_id, subject_id, atom_kind, content, confidence, salience, source_atom_ids, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                atom.atom_id,
                atom.subject_id,
                atom.atom_kind,
                _json_dumps(atom_content_to_dict(atom.content)),
                atom.confidence,
                atom.salience,
                _json_dumps(atom.source_atom_ids),
                atom.created_at,
            ),
        )
        self._commit()
        return atom

    def add_edge(self, edge: MemoryEdge) -> MemoryEdge:
        self.conn.execute(
            """
            INSERT INTO memory_edges (
                edge_id, subject_id, source_atom_id, target_atom_id, influence, confidence, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                edge.edge_id,
                edge.subject_id,
                edge.source_atom_id,
                edge.target_atom_id,
                edge.influence,
                edge.confidence,
                edge.created_at,
            ),
        )
        self._commit()
        return edge

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
    ) -> tuple[MemoryAtom, ...]:
        sql = "SELECT * FROM memory_atoms WHERE subject_id = ?"
        params: list[object] = [subject_id]
        if atom_kinds:
            values = tuple(atom_kinds)
            placeholders = ", ".join("?" for _ in values)
            sql += f" AND atom_kind IN ({placeholders})"
            params.extend(values)
        sql += " ORDER BY created_at ASC, atom_id ASC"
        rows = self.conn.execute(sql, params).fetchall()
        return tuple(self._atom_from_row(row) for row in rows)

    def list_edges(
        self,
        subject_id: str,
        *,
        atom_ids: Iterable[str] | None = None,
    ) -> tuple[MemoryEdge, ...]:
        sql = "SELECT * FROM memory_edges WHERE subject_id = ?"
        params: list[object] = [subject_id]
        if atom_ids:
            values = tuple(atom_ids)
            placeholders = ", ".join("?" for _ in values)
            sql += f" AND (source_atom_id IN ({placeholders}) OR target_atom_id IN ({placeholders}))"
            params.extend(values)
            params.extend(values)
        sql += " ORDER BY created_at ASC, edge_id ASC"
        rows = self.conn.execute(sql, params).fetchall()
        return tuple(self._edge_from_row(row) for row in rows)

    def replace_activation_cache(
        self,
        subject_id: str,
        activations: Mapping[str, float],
        *,
        updated_at: float,
    ) -> None:
        with self.transaction():
            self.conn.execute(
                "DELETE FROM activation_cache WHERE subject_id = ?",
                (subject_id,),
            )
            self.conn.executemany(
                """
                INSERT INTO activation_cache (subject_id, atom_id, activation, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (subject_id, atom_id, float(activation), updated_at)
                    for atom_id, activation in activations.items()
                ],
            )

    def list_activation_cache(self, subject_id: str) -> dict[str, float]:
        rows = self.conn.execute(
            """
            SELECT atom_id, activation
            FROM activation_cache
            WHERE subject_id = ?
            ORDER BY activation DESC, atom_id ASC
            """,
            (subject_id,),
        ).fetchall()
        return {str(row["atom_id"]): float(row["activation"]) for row in rows}

    def _atom_from_row(self, row: sqlite3.Row) -> MemoryAtom:
        atom_kind = atom_kind_from_value(row["atom_kind"])
        return MemoryAtom(
            atom_id=str(row["atom_id"]),
            subject_id=str(row["subject_id"]),
            atom_kind=atom_kind,
            content=atom_content_from_dict(atom_kind, _json_dict(str(row["content"]))),
            confidence=float(row["confidence"]),
            salience=float(row["salience"]),
            source_atom_ids=tuple(_json_list(str(row["source_atom_ids"]))),
            created_at=float(row["created_at"]),
        )

    def _edge_from_row(self, row: sqlite3.Row) -> MemoryEdge:
        return MemoryEdge(
            edge_id=str(row["edge_id"]),
            subject_id=str(row["subject_id"]),
            source_atom_id=str(row["source_atom_id"]),
            target_atom_id=str(row["target_atom_id"]),
            influence=float(row["influence"]),
            confidence=float(row["confidence"]),
            created_at=float(row["created_at"]),
        )

    def _commit(self) -> None:
        if self._tx_depth == 0:
            self.conn.commit()
