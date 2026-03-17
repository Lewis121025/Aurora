"""Aurora v2 SQLite persistence。"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
import sqlite3
from typing import Any, Iterable, Iterator, cast
from uuid import uuid4

import numpy as np

from aurora.runtime.contracts import (
    EventKind,
    EventRecord,
    FactRecord,
    FactStatus,
    LoopStatus,
    LoopType,
    OpenLoop,
    RelationField,
)


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
    """Evidence log、关系场、open loop 与事实的统一 SQLite 存储。"""

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
                session_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at REAL NOT NULL,
                pending_compile INTEGER NOT NULL,
                compiled_at REAL
            );
            CREATE INDEX IF NOT EXISTS idx_events_relation_created
            ON events(relation_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_events_pending
            ON events(relation_id, pending_compile, created_at);

            CREATE TABLE IF NOT EXISTS relation_fields (
                relation_id TEXT PRIMARY KEY,
                trust REAL NOT NULL,
                distance REAL NOT NULL,
                warmth REAL NOT NULL,
                tension REAL NOT NULL,
                repair_debt REAL NOT NULL,
                shared_lexicon TEXT NOT NULL,
                interaction_rules TEXT NOT NULL,
                last_compiled_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS open_loops (
                loop_id TEXT PRIMARY KEY,
                relation_id TEXT NOT NULL,
                loop_type TEXT NOT NULL,
                status TEXT NOT NULL,
                summary TEXT NOT NULL,
                urgency REAL NOT NULL,
                opened_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                evidence_refs TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_loops_relation_status
            ON open_loops(relation_id, status, updated_at);

            CREATE TABLE IF NOT EXISTS facts (
                fact_id TEXT PRIMARY KEY,
                relation_id TEXT NOT NULL,
                content TEXT NOT NULL,
                document_date REAL NOT NULL,
                event_date REAL NOT NULL,
                status TEXT NOT NULL,
                supersedes TEXT,
                confidence REAL NOT NULL,
                evidence_refs TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_facts_relation_status
            ON facts(relation_id, status, event_date DESC);

            CREATE TABLE IF NOT EXISTS fact_embeddings (
                fact_id TEXT PRIMARY KEY,
                relation_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                FOREIGN KEY(fact_id) REFERENCES facts(fact_id)
            );
            CREATE INDEX IF NOT EXISTS idx_fact_embeddings_relation
            ON fact_embeddings(relation_id);
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
        row = self.conn.execute("SELECT COUNT(*) AS count FROM relation_fields").fetchone()
        return int(row["count"])

    def ensure_relation_field(self, relation_id: str) -> RelationField:
        field = self.load_relation_field(relation_id)
        if field is not None:
            return field
        created = RelationField(relation_id=relation_id)
        self.save_relation_field(created)
        return created

    def load_relation_field(self, relation_id: str) -> RelationField | None:
        row = self.conn.execute(
            "SELECT * FROM relation_fields WHERE relation_id = ?",
            (relation_id,),
        ).fetchone()
        if row is None:
            return None
        return RelationField(
            relation_id=relation_id,
            trust=float(row["trust"]),
            distance=float(row["distance"]),
            warmth=float(row["warmth"]),
            tension=float(row["tension"]),
            repair_debt=float(row["repair_debt"]),
            shared_lexicon=_json_list(str(row["shared_lexicon"])),
            interaction_rules=_json_list(str(row["interaction_rules"])),
            last_compiled_at=float(row["last_compiled_at"]),
        )

    def save_relation_field(self, field: RelationField) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO relation_fields (
                relation_id, trust, distance, warmth, tension, repair_debt,
                shared_lexicon, interaction_rules, last_compiled_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                field.relation_id,
                field.trust,
                field.distance,
                field.warmth,
                field.tension,
                field.repair_debt,
                _json_dumps(field.shared_lexicon),
                _json_dumps(field.interaction_rules),
                field.last_compiled_at,
            ),
        )
        self._commit()

    def append_event(
        self,
        *,
        relation_id: str,
        session_id: str,
        kind: EventKind,
        role: str,
        text: str,
        created_at: float,
        payload: dict[str, object] | None = None,
        pending_compile: bool = False,
    ) -> EventRecord:
        event = EventRecord(
            event_id=f"evt_{uuid4().hex[:12]}",
            relation_id=relation_id,
            session_id=session_id,
            kind=kind,
            role=role,
            text=text,
            created_at=created_at,
            pending_compile=pending_compile,
            payload=payload or {},
        )
        self.conn.execute(
            """
            INSERT INTO events (
                event_id, relation_id, session_id, kind, role, text, payload, created_at,
                pending_compile, compiled_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                event.event_id,
                event.relation_id,
                event.session_id,
                event.kind,
                event.role,
                event.text,
                _json_dumps(event.payload),
                event.created_at,
                int(event.pending_compile),
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

    def pending_relation_ids(self) -> tuple[str, ...]:
        rows = self.conn.execute(
            """
            SELECT DISTINCT relation_id
            FROM events
            WHERE kind IN ('user_turn', 'assistant_turn') AND pending_compile = 1
            ORDER BY relation_id
            """
        ).fetchall()
        return tuple(str(row["relation_id"]) for row in rows)

    def pending_turns(self, relation_id: str) -> tuple[EventRecord, ...]:
        rows = self.conn.execute(
            """
            SELECT * FROM events
            WHERE relation_id = ? AND kind IN ('user_turn', 'assistant_turn') AND pending_compile = 1
            ORDER BY created_at
            """,
            (relation_id,),
        ).fetchall()
        return tuple(self._event_from_row(row) for row in rows)

    def mark_events_compiled(self, event_ids: Iterable[str], compiled_at: float) -> None:
        ids = tuple(event_ids)
        if not ids:
            return
        placeholders = ", ".join("?" for _ in ids)
        self.conn.execute(
            f"""
            UPDATE events
            SET pending_compile = 0, compiled_at = ?
            WHERE event_id IN ({placeholders})
            """,
            (compiled_at, *ids),
        )
        self._commit()

    def pending_compile_count(self, relation_id: str) -> int:
        row = self.conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM events
            WHERE relation_id = ? AND kind IN ('user_turn', 'assistant_turn') AND pending_compile = 1
            """,
            (relation_id,),
        ).fetchone()
        return int(row["count"])

    def list_open_loops(self, relation_id: str) -> tuple[OpenLoop, ...]:
        rows = self.conn.execute(
            "SELECT * FROM open_loops WHERE relation_id = ? ORDER BY updated_at DESC",
            (relation_id,),
        ).fetchall()
        return tuple(self._loop_from_row(row) for row in rows)

    def find_open_loop(self, relation_id: str, loop_type: str, summary: str) -> OpenLoop | None:
        row = self.conn.execute(
            """
            SELECT * FROM open_loops
            WHERE relation_id = ? AND loop_type = ? AND summary = ? AND status = 'active'
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (relation_id, loop_type, summary),
        ).fetchone()
        return None if row is None else self._loop_from_row(row)

    def upsert_open_loop(self, loop: OpenLoop) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO open_loops (
                loop_id, relation_id, loop_type, status, summary, urgency,
                opened_at, updated_at, evidence_refs
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                loop.loop_id,
                loop.relation_id,
                loop.loop_type,
                loop.status,
                loop.summary,
                loop.urgency,
                loop.opened_at,
                loop.updated_at,
                _json_dumps(loop.evidence_refs),
            ),
        )
        self._commit()

    def resolve_open_loop(
        self,
        relation_id: str,
        *,
        loop_id: str | None = None,
        summary: str | None = None,
        loop_type: str | None = None,
        updated_at: float,
        evidence_refs: tuple[str, ...] = (),
    ) -> bool:
        loop = None
        if loop_id:
            row = self.conn.execute(
                "SELECT * FROM open_loops WHERE relation_id = ? AND loop_id = ?",
                (relation_id, loop_id),
            ).fetchone()
            loop = None if row is None else self._loop_from_row(row)
        elif summary is not None:
            row = self.conn.execute(
                """
                SELECT * FROM open_loops
                WHERE relation_id = ? AND summary = ? AND status = 'active'
                AND (? IS NULL OR loop_type = ?)
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (relation_id, summary, loop_type, loop_type),
            ).fetchone()
            loop = None if row is None else self._loop_from_row(row)
        if loop is None:
            return False

        merged_refs = tuple(dict.fromkeys((*loop.evidence_refs, *evidence_refs)))
        resolved = OpenLoop(
            loop_id=loop.loop_id,
            relation_id=loop.relation_id,
            loop_type=loop.loop_type,
            status="resolved",
            summary=loop.summary,
            urgency=loop.urgency,
            opened_at=loop.opened_at,
            updated_at=updated_at,
            evidence_refs=merged_refs,
        )
        self.upsert_open_loop(resolved)
        return True

    def list_facts(self, relation_id: str) -> tuple[FactRecord, ...]:
        rows = self.conn.execute(
            "SELECT * FROM facts WHERE relation_id = ? ORDER BY document_date DESC",
            (relation_id,),
        ).fetchall()
        return tuple(self._fact_from_row(row) for row in rows)

    def get_fact(self, fact_id: str) -> FactRecord | None:
        row = self.conn.execute(
            "SELECT * FROM facts WHERE fact_id = ?",
            (fact_id,),
        ).fetchone()
        return None if row is None else self._fact_from_row(row)

    def add_fact(self, record: FactRecord, embedding: np.ndarray) -> None:
        self.conn.execute(
            """
            INSERT INTO facts (
                fact_id, relation_id, content, document_date, event_date,
                status, supersedes, confidence, evidence_refs
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.fact_id,
                record.relation_id,
                record.content,
                record.document_date,
                record.event_date,
                record.status,
                record.supersedes,
                record.confidence,
                _json_dumps(record.evidence_refs),
            ),
        )
        self.conn.execute(
            """
            INSERT OR REPLACE INTO fact_embeddings (fact_id, relation_id, embedding)
            VALUES (?, ?, ?)
            """,
            (record.fact_id, record.relation_id, embedding.astype(np.float32).tobytes()),
        )
        self._commit()

    def update_fact_status(self, fact_id: str, status: str) -> None:
        self.conn.execute(
            "UPDATE facts SET status = ? WHERE fact_id = ?",
            (status, fact_id),
        )
        self._commit()

    def fact_embeddings(self, relation_id: str) -> tuple[tuple[FactRecord, np.ndarray], ...]:
        rows = self.conn.execute(
            """
            SELECT f.*, fe.embedding
            FROM facts AS f
            JOIN fact_embeddings AS fe ON fe.fact_id = f.fact_id
            WHERE f.relation_id = ?
            ORDER BY f.document_date DESC
            """,
            (relation_id,),
        ).fetchall()
        items: list[tuple[FactRecord, np.ndarray]] = []
        for row in rows:
            fact = self._fact_from_row(row)
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            items.append((fact, emb))
        return tuple(items)

    def event_rows_for_recall(self, relation_id: str) -> tuple[EventRecord, ...]:
        rows = self.conn.execute(
            """
            SELECT * FROM events
            WHERE relation_id = ? AND kind IN ('user_turn', 'assistant_turn')
            ORDER BY created_at DESC
            LIMIT 200
            """,
            (relation_id,),
        ).fetchall()
        return tuple(self._event_from_row(row) for row in rows)

    def user_turn_texts(self, relation_id: str) -> tuple[str, ...]:
        rows = self.conn.execute(
            """
            SELECT text FROM events
            WHERE relation_id = ? AND kind = 'user_turn'
            ORDER BY created_at
            """,
            (relation_id,),
        ).fetchall()
        return tuple(str(row["text"]) for row in rows)

    def _event_from_row(self, row: sqlite3.Row) -> EventRecord:
        return EventRecord(
            event_id=str(row["event_id"]),
            relation_id=str(row["relation_id"]),
            session_id=str(row["session_id"]),
            kind=cast(EventKind, str(row["kind"])),
            role=str(row["role"]),
            text=str(row["text"]),
            created_at=float(row["created_at"]),
            pending_compile=bool(row["pending_compile"]),
            payload=_json_dict(str(row["payload"])),
        )

    def _loop_from_row(self, row: sqlite3.Row) -> OpenLoop:
        return OpenLoop(
            loop_id=str(row["loop_id"]),
            relation_id=str(row["relation_id"]),
            loop_type=cast(LoopType, str(row["loop_type"])),
            status=cast(LoopStatus, str(row["status"])),
            summary=str(row["summary"]),
            urgency=float(row["urgency"]),
            opened_at=float(row["opened_at"]),
            updated_at=float(row["updated_at"]),
            evidence_refs=tuple(_json_list(str(row["evidence_refs"]))),
        )

    def _fact_from_row(self, row: sqlite3.Row) -> FactRecord:
        return FactRecord(
            fact_id=str(row["fact_id"]),
            relation_id=str(row["relation_id"]),
            content=str(row["content"]),
            document_date=float(row["document_date"]),
            event_date=float(row["event_date"]),
            status=cast(FactStatus, str(row["status"])),
            supersedes=None if row["supersedes"] is None else str(row["supersedes"]),
            confidence=float(row["confidence"]),
            evidence_refs=tuple(_json_list(str(row["evidence_refs"]))),
        )

    def _commit(self) -> None:
        if self._tx_depth == 0:
            self.conn.commit()
