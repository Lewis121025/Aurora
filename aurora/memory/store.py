"""Aurora SQLite persistence for the expert-scoped memory kernel."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
import re
import sqlite3
from typing import Any, cast

from aurora.memory.experts import GLOBAL_EXPERT_ID, ExpertKind, ExpertRecord
from aurora.runtime.contracts import (
    AtomKind,
    MemoryAtom,
    MemoryEdge,
    TranscriptItem,
    atom_text,
    atom_content_from_dict,
    atom_content_to_dict,
    atom_kind_from_value,
    transcript_role_from_value,
)

_SCHEMA_VERSION = "field-v6"
_BOOTSTRAP_TOPIC_EXPERT_ID = "expert_topic_bootstrap"
_TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+")
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True, slots=True)
class SessionRecord:
    subject_id: str
    session_id: str
    started_at: float
    finalized_at: float | None


@dataclass(frozen=True, slots=True)
class SessionSegmentRecord:
    subject_id: str
    session_id: str
    segment_id: int
    token_count: int
    started_at: float
    closed_at: float | None
    closed_reason: str | None


@dataclass(frozen=True, slots=True)
class SessionEventRecord:
    subject_id: str
    session_id: str
    segment_id: int
    ordinal: int
    atom_id: str


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


def _json_float_tuple(raw: str) -> tuple[float, ...]:
    data = _json_loads(raw)
    if not isinstance(data, list):
        raise ValueError("expected JSON array")
    values: list[float] = []
    for item in data:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError("centroid must contain numeric values")
        values.append(float(item))
    return tuple(values)


class SQLiteMemoryStore:
    """SQLite store for immutable facts plus expert-scoped derived state."""

    def __init__(self, db_path: str) -> None:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._tx_depth = 0
        self._init_schema()

    def _init_schema(self) -> None:
        current_version = self._schema_version()
        with self.transaction():
            self._create_tables()
            self.conn.execute("DROP TABLE IF EXISTS activation_cache")
            if current_version != _SCHEMA_VERSION:
                self._reset_expert_state()
                self._bootstrap_expert_state()
            self._validate_expert_state()
            self.conn.execute(
                """
                INSERT INTO metadata(key, value) VALUES ('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (_SCHEMA_VERSION,),
            )

    def _create_tables(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_atoms (
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
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_atoms_subject_kind
            ON memory_atoms(subject_id, atom_kind, created_at DESC)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_atoms_subject_created
            ON memory_atoms(subject_id, created_at DESC)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_edges (
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
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_edges_subject_source
            ON memory_edges(subject_id, source_atom_id, created_at DESC)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_edges_subject_target
            ON memory_edges(subject_id, target_atom_id, created_at DESC)
            """
        )
        self.conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS reject_memory_edges_touching_evidence
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
            END
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experts (
                subject_id TEXT NOT NULL CHECK(subject_id <> ''),
                expert_id TEXT NOT NULL CHECK(expert_id <> ''),
                expert_kind TEXT NOT NULL CHECK(expert_kind IN ('global', 'topic')),
                centroid TEXT NOT NULL,
                atom_count INTEGER NOT NULL CHECK(atom_count >= 0),
                updated_at REAL NOT NULL,
                last_activated_at REAL NOT NULL,
                PRIMARY KEY(subject_id, expert_id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_experts_single_global
            ON experts(subject_id, expert_kind)
            WHERE expert_kind = 'global'
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_experts_subject_kind_updated
            ON experts(subject_id, expert_kind, updated_at DESC)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS expert_atoms (
                subject_id TEXT NOT NULL CHECK(subject_id <> ''),
                expert_id TEXT NOT NULL CHECK(expert_id <> ''),
                atom_id TEXT NOT NULL CHECK(atom_id <> ''),
                membership_role TEXT NOT NULL CHECK(membership_role IN ('primary', 'secondary')),
                PRIMARY KEY(subject_id, expert_id, atom_id),
                FOREIGN KEY(subject_id, expert_id) REFERENCES experts(subject_id, expert_id) ON DELETE CASCADE,
                FOREIGN KEY(subject_id, atom_id) REFERENCES memory_atoms(subject_id, atom_id) ON DELETE CASCADE
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_expert_atoms_subject_atom
            ON expert_atoms(subject_id, atom_id, membership_role)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_expert_atoms_subject_expert
            ON expert_atoms(subject_id, expert_id, membership_role)
            """
        )
        self.conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_expert_atoms_single_primary
            ON expert_atoms(subject_id, atom_id)
            WHERE membership_role = 'primary'
            """
        )
        self.conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS reject_non_global_secondary_membership_insert
            BEFORE INSERT ON expert_atoms
            FOR EACH ROW
            WHEN NEW.membership_role = 'secondary'
             AND NOT EXISTS (
                SELECT 1
                FROM experts
                WHERE subject_id = NEW.subject_id
                  AND expert_id = NEW.expert_id
                  AND expert_kind = 'global'
            )
            BEGIN
                SELECT RAISE(ABORT, 'secondary memberships require a global expert');
            END
            """
        )
        self.conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS reject_non_global_secondary_membership_update
            BEFORE UPDATE OF expert_id, membership_role ON expert_atoms
            FOR EACH ROW
            WHEN NEW.membership_role = 'secondary'
             AND NOT EXISTS (
                SELECT 1
                FROM experts
                WHERE subject_id = NEW.subject_id
                  AND expert_id = NEW.expert_id
                  AND expert_kind = 'global'
            )
            BEGIN
                SELECT RAISE(ABORT, 'secondary memberships require a global expert');
            END
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS expert_terms (
                subject_id TEXT NOT NULL CHECK(subject_id <> ''),
                expert_id TEXT NOT NULL CHECK(expert_id <> ''),
                token TEXT NOT NULL CHECK(token <> ''),
                weight REAL NOT NULL CHECK(weight >= 0.0),
                PRIMARY KEY(subject_id, expert_id, token),
                FOREIGN KEY(subject_id, expert_id) REFERENCES experts(subject_id, expert_id) ON DELETE CASCADE
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_expert_terms_subject_token
            ON expert_terms(subject_id, token)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS expert_activation_cache (
                subject_id TEXT NOT NULL CHECK(subject_id <> ''),
                expert_id TEXT NOT NULL CHECK(expert_id <> ''),
                atom_id TEXT NOT NULL CHECK(atom_id <> ''),
                activation REAL NOT NULL CHECK(activation BETWEEN 0.0 AND 1.0),
                updated_at REAL NOT NULL,
                PRIMARY KEY(subject_id, expert_id, atom_id),
                FOREIGN KEY(subject_id, expert_id) REFERENCES experts(subject_id, expert_id) ON DELETE CASCADE,
                FOREIGN KEY(subject_id, atom_id) REFERENCES memory_atoms(subject_id, atom_id) ON DELETE CASCADE
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_expert_activation_subject_expert
            ON expert_activation_cache(subject_id, expert_id, activation DESC, updated_at DESC)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                subject_id TEXT NOT NULL CHECK(subject_id <> ''),
                session_id TEXT NOT NULL CHECK(session_id <> ''),
                started_at REAL NOT NULL,
                finalized_at REAL,
                PRIMARY KEY(subject_id, session_id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_sessions_subject_started
            ON sessions(subject_id, started_at DESC, session_id ASC)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_segments (
                subject_id TEXT NOT NULL CHECK(subject_id <> ''),
                session_id TEXT NOT NULL CHECK(session_id <> ''),
                segment_id INTEGER NOT NULL CHECK(segment_id >= 1),
                token_count INTEGER NOT NULL CHECK(token_count >= 0),
                started_at REAL NOT NULL,
                closed_at REAL,
                closed_reason TEXT CHECK(closed_reason IN ('finalize', 'token_cap', 'ingest')),
                PRIMARY KEY(subject_id, session_id, segment_id),
                FOREIGN KEY(subject_id, session_id) REFERENCES sessions(subject_id, session_id) ON DELETE CASCADE
            )
            """
        )
        self.conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_session_segments_single_open
            ON session_segments(subject_id, session_id)
            WHERE closed_at IS NULL
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_session_segments_subject_started
            ON session_segments(subject_id, session_id, segment_id DESC)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_events (
                subject_id TEXT NOT NULL CHECK(subject_id <> ''),
                session_id TEXT NOT NULL CHECK(session_id <> ''),
                segment_id INTEGER NOT NULL CHECK(segment_id >= 1),
                ordinal INTEGER NOT NULL CHECK(ordinal >= 1),
                atom_id TEXT NOT NULL CHECK(atom_id <> ''),
                PRIMARY KEY(subject_id, session_id, segment_id, ordinal),
                UNIQUE(subject_id, session_id, segment_id, atom_id),
                FOREIGN KEY(subject_id, session_id, segment_id)
                    REFERENCES session_segments(subject_id, session_id, segment_id)
                    ON DELETE CASCADE,
                FOREIGN KEY(subject_id, atom_id)
                    REFERENCES memory_atoms(subject_id, atom_id)
                    ON DELETE CASCADE
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_session_events_subject_atom
            ON session_events(subject_id, atom_id)
            """
        )

    def _schema_version(self) -> str | None:
        metadata_exists = self.conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'metadata'
            """
        ).fetchone()
        if metadata_exists is None:
            return None
        row = self.conn.execute(
            """
            SELECT value
            FROM metadata
            WHERE key = 'schema_version'
            """
        ).fetchone()
        return None if row is None else str(row["value"])

    def _reset_expert_state(self) -> None:
        self.conn.execute("DELETE FROM expert_activation_cache")
        self.conn.execute("DELETE FROM expert_terms")
        self.conn.execute("DELETE FROM expert_atoms")
        self.conn.execute("DELETE FROM experts")

    def _bootstrap_expert_state(self) -> None:
        subject_rows = self.conn.execute(
            """
            SELECT subject_id, COALESCE(MAX(created_at), 0.0) AS latest_created_at
            FROM memory_atoms
            GROUP BY subject_id
            ORDER BY subject_id ASC
            """
        ).fetchall()
        for row in subject_rows:
            subject_id = str(row["subject_id"])
            updated_at = float(row["latest_created_at"])
            self.conn.execute(
                """
                INSERT INTO experts (
                    subject_id, expert_id, expert_kind, centroid, atom_count, updated_at, last_activated_at
                ) VALUES (?, ?, 'global', '[]', ?, ?, 0.0)
                """,
                (subject_id, GLOBAL_EXPERT_ID, 0, updated_at),
            )
            non_evidence_ids = tuple(
                str(atom_row["atom_id"])
                for atom_row in self.conn.execute(
                    """
                    SELECT atom_id
                    FROM memory_atoms
                    WHERE subject_id = ?
                      AND atom_kind <> 'evidence'
                    ORDER BY created_at ASC, atom_id ASC
                    """,
                    (subject_id,),
                ).fetchall()
            )
            if non_evidence_ids:
                self.conn.execute(
                    """
                    INSERT INTO experts (
                        subject_id, expert_id, expert_kind, centroid, atom_count, updated_at, last_activated_at
                    ) VALUES (?, ?, 'topic', '[]', ?, ?, 0.0)
                    """,
                    (subject_id, _BOOTSTRAP_TOPIC_EXPERT_ID, len(non_evidence_ids), updated_at),
                )
                self.conn.executemany(
                    """
                    INSERT INTO expert_atoms (subject_id, expert_id, atom_id, membership_role)
                    VALUES (?, ?, ?, 'primary')
                    """,
                    [(subject_id, _BOOTSTRAP_TOPIC_EXPERT_ID, atom_id) for atom_id in non_evidence_ids],
                )
                secondary_ids = self._bootstrap_global_atom_ids(subject_id)
                if secondary_ids:
                    self.conn.executemany(
                        """
                        INSERT INTO expert_atoms (subject_id, expert_id, atom_id, membership_role)
                        VALUES (?, ?, ?, 'secondary')
                        """,
                        [(subject_id, GLOBAL_EXPERT_ID, atom_id) for atom_id in secondary_ids],
                    )
                    self.conn.execute(
                        """
                        UPDATE experts
                        SET atom_count = ?
                        WHERE subject_id = ?
                          AND expert_id = ?
                        """,
                        (len(secondary_ids), subject_id, GLOBAL_EXPERT_ID),
                    )
                term_weights = self._bootstrap_topic_terms(subject_id)
                if term_weights:
                    self.conn.executemany(
                        """
                        INSERT INTO expert_terms (subject_id, expert_id, token, weight)
                        VALUES (?, ?, ?, ?)
                        """,
                        [
                            (subject_id, _BOOTSTRAP_TOPIC_EXPERT_ID, token, weight)
                            for token, weight in sorted(term_weights.items())
                        ],
                    )

    def _bootstrap_global_atom_ids(self, subject_id: str) -> tuple[str, ...]:
        atom_ids: list[str] = []
        atom_ids.extend(self._latest_atom_ids(subject_id, "episode", limit=6))
        atom_ids.extend(self._latest_atom_ids(subject_id, "memory", limit=6))
        atom_ids.extend(self._latest_atom_ids(subject_id, "inhibition", limit=4))
        return tuple(dict.fromkeys(atom_ids))

    def _latest_atom_ids(self, subject_id: str, atom_kind: AtomKind, *, limit: int) -> tuple[str, ...]:
        rows = self.conn.execute(
            """
            SELECT atom_id
            FROM memory_atoms
            WHERE subject_id = ?
              AND atom_kind = ?
            ORDER BY created_at DESC, atom_id DESC
            LIMIT ?
            """,
            (subject_id, atom_kind, limit),
        ).fetchall()
        return tuple(str(row["atom_id"]) for row in rows)

    def _bootstrap_topic_terms(self, subject_id: str) -> dict[str, float]:
        weights: dict[str, float] = {}
        for atom in self.list_atoms(subject_id):
            if atom.atom_kind == "evidence":
                continue
            for token in _tokenize(atom_text(atom)):
                weights[token] = weights.get(token, 0.0) + 1.0
        return weights

    def _validate_expert_state(self) -> None:
        broken_subjects: list[str] = []
        subjects = tuple(
            str(row["subject_id"])
            for row in self.conn.execute(
                """
                SELECT subject_id FROM experts
                UNION
                SELECT subject_id FROM memory_atoms
                ORDER BY subject_id ASC
                """
            ).fetchall()
        )
        for subject_id in subjects:
            if not self._subject_expert_state_is_complete(subject_id):
                broken_subjects.append(subject_id)
        if broken_subjects:
            joined = ", ".join(broken_subjects)
            raise RuntimeError(f"incomplete expert state for subjects: {joined}")

    def _subject_expert_state_is_complete(self, subject_id: str) -> bool:
        fact_count_row = self.conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM memory_atoms
            WHERE subject_id = ?
            """,
            (subject_id,),
        ).fetchone()
        fact_count = int(fact_count_row["count"] or 0)
        non_evidence_count_row = self.conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM memory_atoms
            WHERE subject_id = ?
              AND atom_kind <> 'evidence'
            """,
            (subject_id,),
        ).fetchone()
        non_evidence_count = int(non_evidence_count_row["count"] or 0)
        counts = self.conn.execute(
            """
            SELECT
                SUM(CASE WHEN expert_kind = 'global' THEN 1 ELSE 0 END) AS global_count,
                SUM(CASE WHEN expert_kind = 'topic' THEN 1 ELSE 0 END) AS topic_count
            FROM experts
            WHERE subject_id = ?
            """,
            (subject_id,),
        ).fetchone()
        global_count = int(counts["global_count"] or 0)
        topic_count = int(counts["topic_count"] or 0)
        if global_count == 0 and topic_count == 0:
            return fact_count == 0
        if global_count != 1:
            return False
        if non_evidence_count > 0 and topic_count < 1:
            return False
        missing_primary = self.conn.execute(
            """
            SELECT 1
            FROM memory_atoms AS a
            WHERE a.subject_id = ?
              AND a.atom_kind <> 'evidence'
              AND (
                SELECT COUNT(*)
                FROM expert_atoms AS m
                WHERE m.subject_id = a.subject_id
                  AND m.atom_id = a.atom_id
                  AND m.membership_role = 'primary'
              ) <> 1
            LIMIT 1
            """,
            (subject_id,),
        ).fetchone()
        if missing_primary is not None:
            return False
        invalid_secondary = self.conn.execute(
            """
            SELECT 1
            FROM expert_atoms AS m
            JOIN experts AS e
              ON e.subject_id = m.subject_id
             AND e.expert_id = m.expert_id
            WHERE m.subject_id = ?
              AND m.membership_role = 'secondary'
              AND e.expert_kind <> 'global'
            LIMIT 1
            """,
            (subject_id,),
        ).fetchone()
        return invalid_secondary is None

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

    def list_subject_ids(self) -> tuple[str, ...]:
        rows = self.conn.execute(
            """
            SELECT DISTINCT subject_id
            FROM memory_atoms
            ORDER BY subject_id ASC
            """
        ).fetchall()
        return tuple(str(row["subject_id"]) for row in rows)

    def get_session(self, subject_id: str, session_id: str) -> SessionRecord | None:
        row = self.conn.execute(
            """
            SELECT *
            FROM sessions
            WHERE subject_id = ?
              AND session_id = ?
            """,
            (subject_id, session_id),
        ).fetchone()
        return None if row is None else self._session_from_row(row)

    def ensure_session(self, subject_id: str, session_id: str, *, started_at: float) -> SessionRecord:
        existing = self.get_session(subject_id, session_id)
        if existing is not None:
            return existing
        self.conn.execute(
            """
            INSERT INTO sessions (subject_id, session_id, started_at, finalized_at)
            VALUES (?, ?, ?, NULL)
            """,
            (subject_id, session_id, started_at),
        )
        self._commit()
        return SessionRecord(subject_id=subject_id, session_id=session_id, started_at=started_at, finalized_at=None)

    def close_session(self, subject_id: str, session_id: str, *, finalized_at: float) -> None:
        self.conn.execute(
            """
            UPDATE sessions
            SET finalized_at = ?
            WHERE subject_id = ?
              AND session_id = ?
            """,
            (finalized_at, subject_id, session_id),
        )
        self._commit()

    def get_open_segment(self, subject_id: str, session_id: str) -> SessionSegmentRecord | None:
        row = self.conn.execute(
            """
            SELECT *
            FROM session_segments
            WHERE subject_id = ?
              AND session_id = ?
              AND closed_at IS NULL
            LIMIT 1
            """,
            (subject_id, session_id),
        ).fetchone()
        return None if row is None else self._segment_from_row(row)

    def create_segment(self, subject_id: str, session_id: str, *, started_at: float) -> SessionSegmentRecord:
        row = self.conn.execute(
            """
            SELECT COALESCE(MAX(segment_id), 0) + 1 AS next_segment_id
            FROM session_segments
            WHERE subject_id = ?
              AND session_id = ?
            """,
            (subject_id, session_id),
        ).fetchone()
        segment_id = int(row["next_segment_id"])
        self.conn.execute(
            """
            INSERT INTO session_segments (
                subject_id, session_id, segment_id, token_count, started_at, closed_at, closed_reason
            ) VALUES (?, ?, ?, 0, ?, NULL, NULL)
            """,
            (subject_id, session_id, segment_id, started_at),
        )
        self._commit()
        return SessionSegmentRecord(
            subject_id=subject_id,
            session_id=session_id,
            segment_id=segment_id,
            token_count=0,
            started_at=started_at,
            closed_at=None,
            closed_reason=None,
        )

    def ensure_open_segment(self, subject_id: str, session_id: str, *, started_at: float) -> SessionSegmentRecord:
        segment = self.get_open_segment(subject_id, session_id)
        if segment is not None:
            return segment
        return self.create_segment(subject_id, session_id, started_at=started_at)

    def close_segment(
        self,
        subject_id: str,
        session_id: str,
        segment_id: int,
        *,
        closed_at: float,
        closed_reason: str,
    ) -> None:
        self.conn.execute(
            """
            UPDATE session_segments
            SET closed_at = ?, closed_reason = ?
            WHERE subject_id = ?
              AND session_id = ?
              AND segment_id = ?
              AND closed_at IS NULL
            """,
            (closed_at, closed_reason, subject_id, session_id, segment_id),
        )
        self._commit()

    def append_session_event(
        self,
        subject_id: str,
        session_id: str,
        segment_id: int,
        atom_id: str,
        *,
        token_count_increment: int,
    ) -> SessionEventRecord:
        row = self.conn.execute(
            """
            SELECT COALESCE(MAX(ordinal), 0) + 1 AS next_ordinal
            FROM session_events
            WHERE subject_id = ?
              AND session_id = ?
              AND segment_id = ?
            """,
            (subject_id, session_id, segment_id),
        ).fetchone()
        ordinal = int(row["next_ordinal"])
        self.conn.execute(
            """
            INSERT INTO session_events (subject_id, session_id, segment_id, ordinal, atom_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (subject_id, session_id, segment_id, ordinal, atom_id),
        )
        if token_count_increment > 0:
            self.conn.execute(
                """
                UPDATE session_segments
                SET token_count = token_count + ?
                WHERE subject_id = ?
                  AND session_id = ?
                  AND segment_id = ?
                """,
                (token_count_increment, subject_id, session_id, segment_id),
            )
        self._commit()
        return SessionEventRecord(
            subject_id=subject_id,
            session_id=session_id,
            segment_id=segment_id,
            ordinal=ordinal,
            atom_id=atom_id,
        )

    def list_segment_transcript(
        self,
        subject_id: str,
        session_id: str,
        segment_id: int,
    ) -> tuple[TranscriptItem, ...]:
        rows = self.conn.execute(
            """
            SELECT e.ordinal, a.created_at, a.content
            FROM session_events AS e
            JOIN memory_atoms AS a
              ON a.subject_id = e.subject_id
             AND a.atom_id = e.atom_id
            WHERE e.subject_id = ?
              AND e.session_id = ?
              AND e.segment_id = ?
            ORDER BY e.ordinal ASC
            """,
            (subject_id, session_id, segment_id),
        ).fetchall()
        transcript: list[TranscriptItem] = []
        for row in rows:
            payload = _json_dict(str(row["content"]))
            role = transcript_role_from_value(payload.get("role"))
            transcript.append(
                TranscriptItem(
                    role=role,
                    text=_required_content_string(payload, "text"),
                    created_at=float(row["created_at"]),
                )
            )
        return tuple(transcript)

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

    def list_atoms_by_ids(self, subject_id: str, atom_ids: Iterable[str]) -> tuple[MemoryAtom, ...]:
        values = tuple(dict.fromkeys(atom_ids))
        if not values:
            return ()
        placeholders = ", ".join("?" for _ in values)
        rows = self.conn.execute(
            f"""
            SELECT *
            FROM memory_atoms
            WHERE subject_id = ?
              AND atom_id IN ({placeholders})
            ORDER BY created_at ASC, atom_id ASC
            """,
            [subject_id, *values],
        ).fetchall()
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
            values = tuple(dict.fromkeys(atom_ids))
            placeholders = ", ".join("?" for _ in values)
            sql += f" AND (source_atom_id IN ({placeholders}) OR target_atom_id IN ({placeholders}))"
            params.extend(values)
            params.extend(values)
        sql += " ORDER BY created_at ASC, edge_id ASC"
        rows = self.conn.execute(sql, params).fetchall()
        return tuple(self._edge_from_row(row) for row in rows)

    def add_expert(self, record: ExpertRecord) -> ExpertRecord:
        self.conn.execute(
            """
            INSERT INTO experts (
                subject_id, expert_id, expert_kind, centroid, atom_count, updated_at, last_activated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.subject_id,
                record.expert_id,
                record.expert_kind,
                _json_dumps(record.centroid),
                record.atom_count,
                record.updated_at,
                record.last_activated_at,
            ),
        )
        self._commit()
        return record

    def get_expert(self, subject_id: str, expert_id: str) -> ExpertRecord | None:
        row = self.conn.execute(
            """
            SELECT *
            FROM experts
            WHERE subject_id = ?
              AND expert_id = ?
            """,
            (subject_id, expert_id),
        ).fetchone()
        return None if row is None else self._expert_from_row(row)

    def list_experts(
        self,
        subject_id: str,
        *,
        expert_kind: ExpertKind | None = None,
    ) -> tuple[ExpertRecord, ...]:
        sql = "SELECT * FROM experts WHERE subject_id = ?"
        params: list[object] = [subject_id]
        if expert_kind is not None:
            sql += " AND expert_kind = ?"
            params.append(expert_kind)
        sql += " ORDER BY expert_kind ASC, updated_at DESC, expert_id ASC"
        rows = self.conn.execute(sql, params).fetchall()
        return tuple(self._expert_from_row(row) for row in rows)

    def update_expert(
        self,
        subject_id: str,
        expert_id: str,
        *,
        centroid: tuple[float, ...],
        atom_count: int,
        updated_at: float,
    ) -> None:
        self.conn.execute(
            """
            UPDATE experts
            SET centroid = ?, atom_count = ?, updated_at = ?
            WHERE subject_id = ?
              AND expert_id = ?
            """,
            (_json_dumps(centroid), atom_count, updated_at, subject_id, expert_id),
        )
        self._commit()

    def touch_experts(self, subject_id: str, expert_ids: Iterable[str], *, activated_at: float) -> None:
        values = tuple(dict.fromkeys(expert_ids))
        if not values:
            return
        placeholders = ", ".join("?" for _ in values)
        self.conn.execute(
            f"""
            UPDATE experts
            SET last_activated_at = ?
            WHERE subject_id = ?
              AND expert_id IN ({placeholders})
            """,
            [activated_at, subject_id, *values],
        )
        self._commit()

    def delete_expert(self, subject_id: str, expert_id: str) -> None:
        self.conn.execute(
            """
            DELETE FROM experts
            WHERE subject_id = ?
              AND expert_id = ?
            """,
            (subject_id, expert_id),
        )
        self._commit()

    def set_atom_memberships(
        self,
        subject_id: str,
        atom_id: str,
        *,
        primary_expert_id: str,
        secondary_expert_ids: Iterable[str] = (),
    ) -> None:
        secondary_ids = tuple(dict.fromkeys(expert_id for expert_id in secondary_expert_ids if expert_id != primary_expert_id))
        with self.transaction():
            self.conn.execute(
                """
                DELETE FROM expert_atoms
                WHERE subject_id = ?
                  AND atom_id = ?
                """,
                (subject_id, atom_id),
            )
            self.conn.execute(
                """
                INSERT INTO expert_atoms (subject_id, expert_id, atom_id, membership_role)
                VALUES (?, ?, ?, 'primary')
                """,
                (subject_id, primary_expert_id, atom_id),
            )
            if secondary_ids:
                self.conn.executemany(
                    """
                    INSERT INTO expert_atoms (subject_id, expert_id, atom_id, membership_role)
                    VALUES (?, ?, ?, 'secondary')
                    """,
                    [(subject_id, expert_id, atom_id) for expert_id in secondary_ids],
                )

    def replace_secondary_memberships(self, subject_id: str, expert_id: str, atom_ids: Iterable[str]) -> None:
        values = tuple(dict.fromkeys(atom_ids))
        with self.transaction():
            self.conn.execute(
                """
                DELETE FROM expert_atoms
                WHERE subject_id = ?
                  AND expert_id = ?
                  AND membership_role = 'secondary'
                """,
                (subject_id, expert_id),
            )
            if values:
                self.conn.executemany(
                    """
                    INSERT INTO expert_atoms (subject_id, expert_id, atom_id, membership_role)
                    VALUES (?, ?, ?, 'secondary')
                    """,
                    [(subject_id, expert_id, atom_id) for atom_id in values],
                )

    def primary_expert_id(self, subject_id: str, atom_id: str) -> str | None:
        row = self.conn.execute(
            """
            SELECT expert_id
            FROM expert_atoms
            WHERE subject_id = ?
              AND atom_id = ?
              AND membership_role = 'primary'
            LIMIT 1
            """,
            (subject_id, atom_id),
        ).fetchone()
        return None if row is None else str(row["expert_id"])

    def list_expert_atom_ids(
        self,
        subject_id: str,
        expert_id: str,
        *,
        membership_role: str | None = None,
    ) -> tuple[str, ...]:
        sql = """
            SELECT atom_id
            FROM expert_atoms
            WHERE subject_id = ?
              AND expert_id = ?
        """
        params: list[object] = [subject_id, expert_id]
        if membership_role is not None:
            sql += " AND membership_role = ?"
            params.append(membership_role)
        sql += " ORDER BY atom_id ASC"
        rows = self.conn.execute(sql, params).fetchall()
        return tuple(str(row["atom_id"]) for row in rows)

    def list_expert_atoms(
        self,
        subject_id: str,
        expert_id: str,
        *,
        membership_role: str | None = None,
        atom_kinds: Iterable[AtomKind] | None = None,
    ) -> tuple[MemoryAtom, ...]:
        sql = """
            SELECT a.*
            FROM memory_atoms AS a
            JOIN expert_atoms AS m
              ON m.subject_id = a.subject_id
             AND m.atom_id = a.atom_id
            WHERE a.subject_id = ?
              AND m.expert_id = ?
        """
        params: list[object] = [subject_id, expert_id]
        if membership_role is not None:
            sql += " AND m.membership_role = ?"
            params.append(membership_role)
        if atom_kinds:
            values = tuple(atom_kinds)
            placeholders = ", ".join("?" for _ in values)
            sql += f" AND a.atom_kind IN ({placeholders})"
            params.extend(values)
        sql += " ORDER BY a.created_at ASC, a.atom_id ASC"
        rows = self.conn.execute(sql, params).fetchall()
        return tuple(self._atom_from_row(row) for row in rows)

    def replace_expert_terms(
        self,
        subject_id: str,
        expert_id: str,
        weights: Mapping[str, float],
    ) -> None:
        with self.transaction():
            self.conn.execute(
                """
                DELETE FROM expert_terms
                WHERE subject_id = ?
                  AND expert_id = ?
                """,
                (subject_id, expert_id),
            )
            if weights:
                self.conn.executemany(
                    """
                    INSERT INTO expert_terms (subject_id, expert_id, token, weight)
                    VALUES (?, ?, ?, ?)
                    """,
                    [
                        (subject_id, expert_id, token, float(weight))
                        for token, weight in sorted(weights.items())
                        if token and float(weight) > 0.0
                    ],
                )

    def expert_token_overlap(
        self,
        subject_id: str,
        tokens: Iterable[str],
        *,
        expert_kind: ExpertKind = "topic",
    ) -> dict[str, float]:
        values = tuple(dict.fromkeys(token for token in tokens if token))
        if not values:
            return {}
        placeholders = ", ".join("?" for _ in values)
        rows = self.conn.execute(
            f"""
            SELECT t.expert_id, SUM(t.weight) AS overlap
            FROM expert_terms AS t
            JOIN experts AS e
              ON e.subject_id = t.subject_id
             AND e.expert_id = t.expert_id
            WHERE t.subject_id = ?
              AND e.expert_kind = ?
              AND t.token IN ({placeholders})
            GROUP BY t.expert_id
            ORDER BY overlap DESC, t.expert_id ASC
            """,
            [subject_id, expert_kind, *values],
        ).fetchall()
        return {str(row["expert_id"]): float(row["overlap"]) for row in rows}

    def replace_expert_activation_cache(
        self,
        subject_id: str,
        expert_id: str,
        activations: Mapping[str, float],
        *,
        updated_at: float,
    ) -> None:
        with self.transaction():
            self.conn.execute(
                """
                DELETE FROM expert_activation_cache
                WHERE subject_id = ?
                  AND expert_id = ?
                """,
                (subject_id, expert_id),
            )
            if activations:
                self.conn.executemany(
                    """
                    INSERT INTO expert_activation_cache (
                        subject_id, expert_id, atom_id, activation, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (subject_id, expert_id, atom_id, float(activation), updated_at)
                        for atom_id, activation in activations.items()
                    ],
                )

    def list_expert_activation_cache(self, subject_id: str, expert_id: str) -> dict[str, float]:
        rows = self.conn.execute(
            """
            SELECT atom_id, activation
            FROM expert_activation_cache
            WHERE subject_id = ?
              AND expert_id = ?
            ORDER BY activation DESC, atom_id ASC
            """,
            (subject_id, expert_id),
        ).fetchall()
        return {str(row["atom_id"]): float(row["activation"]) for row in rows}

    def list_activation_cache(self, subject_id: str) -> dict[str, float]:
        rows = self.conn.execute(
            """
            SELECT atom_id, MAX(activation) AS activation
            FROM expert_activation_cache
            WHERE subject_id = ?
            GROUP BY atom_id
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

    def _expert_from_row(self, row: sqlite3.Row) -> ExpertRecord:
        expert_kind = str(row["expert_kind"])
        if expert_kind not in {"global", "topic"}:
            raise ValueError(f"invalid expert_kind: {expert_kind}")
        return ExpertRecord(
            expert_id=str(row["expert_id"]),
            subject_id=str(row["subject_id"]),
            expert_kind=cast(ExpertKind, expert_kind),
            centroid=_json_float_tuple(str(row["centroid"])),
            atom_count=int(row["atom_count"]),
            updated_at=float(row["updated_at"]),
            last_activated_at=float(row["last_activated_at"]),
        )

    def _session_from_row(self, row: sqlite3.Row) -> SessionRecord:
        finalized_at = row["finalized_at"]
        return SessionRecord(
            subject_id=str(row["subject_id"]),
            session_id=str(row["session_id"]),
            started_at=float(row["started_at"]),
            finalized_at=None if finalized_at is None else float(finalized_at),
        )

    def _segment_from_row(self, row: sqlite3.Row) -> SessionSegmentRecord:
        closed_at = row["closed_at"]
        closed_reason = row["closed_reason"]
        return SessionSegmentRecord(
            subject_id=str(row["subject_id"]),
            session_id=str(row["session_id"]),
            segment_id=int(row["segment_id"]),
            token_count=int(row["token_count"]),
            started_at=float(row["started_at"]),
            closed_at=None if closed_at is None else float(closed_at),
            closed_reason=None if closed_reason is None else str(closed_reason),
        )

    def _commit(self) -> None:
        if self._tx_depth == 0:
            self.conn.commit()


def _tokenize(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for chunk in _TOKEN_PATTERN.findall(text.lower()):
        if _CJK_PATTERN.search(chunk):
            tokens.extend(char for char in chunk if _CJK_PATTERN.match(char))
            continue
        tokens.append(chunk)
    return tuple(tokens)


def _required_content_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{key} must be non-empty")
    return text
