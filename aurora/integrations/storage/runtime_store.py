from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from aurora.utils.jsonx import dumps, loads

RUNTIME_SYSTEM_SUBJECT_ID = "__runtime__"


@dataclass(frozen=True)
class StoredEvent:
    seq: int
    event_id: str
    event_type: str
    session_id: str
    ts: float
    payload: Dict[str, Any]


@dataclass(frozen=True)
class ProjectionStatus:
    subject_id: str
    subject_type: str
    status: str
    event_seq: Optional[int]
    node_id: Optional[str]
    node_kind: Optional[str]
    error: Optional[str]
    payload: Dict[str, Any]
    updated_ts: float


@dataclass(frozen=True)
class StoredJob:
    job_id: str
    job_type: str
    event_id: Optional[str]
    payload: Dict[str, Any]
    status: str
    attempts: int
    max_attempts: int
    available_at: float
    lease_owner: Optional[str]
    lease_expires_at: Optional[float]
    dedupe_key: Optional[str]
    created_ts: float
    updated_ts: float
    last_error: Optional[str]


@dataclass(frozen=True)
class OverlayHit:
    event_id: str
    session_id: str
    snippet: str
    score: float
    ts: float
    projection_status: str


class SQLiteRuntimeStore:
    """Single-file runtime store for events, queue, overlay search, and projection state."""

    def __init__(self, path: str):
        self.path = path
        self._closed = False
        self._bootstrap()

    def _connect(self) -> sqlite3.Connection:
        if self._closed:
            raise RuntimeError("Runtime store connection is closed")
        conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=3000")
        return conn

    def _bootstrap(self) -> None:
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS events ("
                "seq INTEGER PRIMARY KEY AUTOINCREMENT,"
                "event_id TEXT UNIQUE NOT NULL,"
                "event_type TEXT NOT NULL,"
                "session_id TEXT NOT NULL,"
                "ts REAL NOT NULL,"
                "payload TEXT NOT NULL"
                ")"
            )
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS event_fts USING fts5("
                "event_id UNINDEXED,"
                "session_id UNINDEXED,"
                "search_text"
                ")"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS jobs ("
                "job_id TEXT PRIMARY KEY,"
                "job_type TEXT NOT NULL,"
                "event_id TEXT,"
                "payload TEXT NOT NULL,"
                "status TEXT NOT NULL,"
                "attempts INTEGER NOT NULL DEFAULT 0,"
                "max_attempts INTEGER NOT NULL,"
                "available_at REAL NOT NULL,"
                "lease_owner TEXT,"
                "lease_expires_at REAL,"
                "dedupe_key TEXT UNIQUE,"
                "created_ts REAL NOT NULL,"
                "updated_ts REAL NOT NULL,"
                "last_error TEXT"
                ")"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_claim "
                "ON jobs(status, available_at, lease_expires_at)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS job_attempts ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "job_id TEXT NOT NULL,"
                "attempt_no INTEGER NOT NULL,"
                "status TEXT NOT NULL,"
                "error TEXT,"
                "ts REAL NOT NULL"
                ")"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS projection_state ("
                "subject_id TEXT PRIMARY KEY,"
                "subject_type TEXT NOT NULL,"
                "status TEXT NOT NULL,"
                "event_seq INTEGER,"
                "node_id TEXT,"
                "node_kind TEXT,"
                "error TEXT,"
                "payload TEXT NOT NULL,"
                "updated_ts REAL NOT NULL"
                ")"
            )
            conn.execute(
                "INSERT OR IGNORE INTO projection_state("
                "subject_id, subject_type, status, event_seq, node_id, node_kind, error, payload, updated_ts"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    RUNTIME_SYSTEM_SUBJECT_ID,
                    "system",
                    "idle",
                    0,
                    None,
                    None,
                    None,
                    dumps({"last_projected_seq": 0, "last_fade_ts": None, "last_evolve_ts": None}),
                    time.time(),
                ),
            )
        finally:
            conn.close()

    def _in_tx(self, fn):
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            result = fn(conn)
            conn.execute("COMMIT")
            return result
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def _enqueue_job_row(
        self,
        conn: sqlite3.Connection,
        *,
        job_type: str,
        payload: Dict[str, Any],
        event_id: Optional[str],
        max_attempts: int,
        available_at: float,
        dedupe_key: Optional[str],
    ) -> StoredJob:
        now = time.time()
        existing = None
        if dedupe_key is not None:
            existing = conn.execute(
                "SELECT * FROM jobs WHERE dedupe_key = ?",
                (dedupe_key,),
            ).fetchone()
        if existing is not None:
            return self._row_to_job(existing)
        job_id = f"job_{uuid.uuid4().hex}"
        conn.execute(
            "INSERT INTO jobs("
            "job_id, job_type, event_id, payload, status, attempts, max_attempts, "
            "available_at, lease_owner, lease_expires_at, dedupe_key, created_ts, updated_ts, last_error"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job_id,
                job_type,
                event_id,
                dumps(payload),
                "queued",
                0,
                int(max_attempts),
                float(available_at),
                None,
                None,
                dedupe_key,
                now,
                now,
                None,
            ),
        )
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise RuntimeError(f"Failed to create job {job_id}")
        return self._row_to_job(row)

    def append_event_and_enqueue(
        self,
        *,
        event_id: str,
        event_type: str,
        session_id: str,
        ts: float,
        payload: Dict[str, Any],
        search_text: str,
        job_type: str,
        job_payload: Dict[str, Any],
        max_attempts: int,
        dedupe_key: Optional[str] = None,
    ) -> Tuple[StoredEvent, StoredJob]:
        def _tx(conn: sqlite3.Connection) -> Tuple[StoredEvent, StoredJob]:
            existing = conn.execute(
                "SELECT * FROM events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if existing is not None:
                event = self._row_to_event(existing)
                job = conn.execute("SELECT * FROM jobs WHERE event_id = ? ORDER BY created_ts ASC", (event_id,)).fetchone()
                if job is None:
                    raise RuntimeError(f"Missing job for existing event {event_id}")
                return event, self._row_to_job(job)

            conn.execute(
                "INSERT INTO events(event_id, event_type, session_id, ts, payload) VALUES (?, ?, ?, ?, ?)",
                (event_id, event_type, session_id, float(ts), dumps(payload)),
            )
            event_row = conn.execute(
                "SELECT * FROM events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if event_row is None:
                raise RuntimeError(f"Failed to persist event {event_id}")
            event = self._row_to_event(event_row)
            conn.execute(
                "INSERT INTO projection_state("
                "subject_id, subject_type, status, event_seq, node_id, node_kind, error, payload, updated_ts"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event_id,
                    "event",
                    "accepted",
                    event.seq,
                    None,
                    None,
                    None,
                    dumps({"event_type": event_type}),
                    time.time(),
                ),
            )
            conn.execute(
                "INSERT INTO event_fts(event_id, session_id, search_text) VALUES (?, ?, ?)",
                (event_id, session_id, search_text),
            )
            job = self._enqueue_job_row(
                conn,
                job_type=job_type,
                payload=job_payload,
                event_id=event_id,
                max_attempts=max_attempts,
                available_at=time.time(),
                dedupe_key=dedupe_key,
            )
            return event, job

        return self._in_tx(_tx)

    def enqueue_job(
        self,
        *,
        job_type: str,
        payload: Dict[str, Any],
        event_id: Optional[str] = None,
        max_attempts: int = 3,
        available_at: Optional[float] = None,
        dedupe_key: Optional[str] = None,
    ) -> StoredJob:
        def _tx(conn: sqlite3.Connection) -> StoredJob:
            return self._enqueue_job_row(
                conn,
                job_type=job_type,
                payload=payload,
                event_id=event_id,
                max_attempts=max_attempts,
                available_at=float(time.time() if available_at is None else available_at),
                dedupe_key=dedupe_key,
            )

        return self._in_tx(_tx)

    def iter_events(self, *, after_seq: int = 0) -> Iterable[StoredEvent]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM events WHERE seq > ? ORDER BY seq ASC",
                (int(after_seq),),
            ).fetchall()
        finally:
            conn.close()
        for row in rows:
            yield self._row_to_event(row)

    def iter_projected_events(self, *, after_seq: int = 0) -> Iterable[StoredEvent]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT e.* FROM events e "
                "JOIN projection_state p ON p.subject_id = e.event_id "
                "WHERE e.seq > ? AND p.subject_type = 'event' AND p.status = 'projected' "
                "ORDER BY e.seq ASC",
                (int(after_seq),),
            ).fetchall()
        finally:
            conn.close()
        for row in rows:
            yield self._row_to_event(row)

    def get_event(self, event_id: str) -> Optional[StoredEvent]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        finally:
            conn.close()
        return None if row is None else self._row_to_event(row)

    def get_projection_status(self, subject_id: str) -> Optional[ProjectionStatus]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM projection_state WHERE subject_id = ?",
                (subject_id,),
            ).fetchone()
        finally:
            conn.close()
        return None if row is None else self._row_to_projection(row)

    def mark_projecting(self, event_id: str) -> None:
        def _tx(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE projection_state SET status = ?, error = NULL, updated_ts = ? WHERE subject_id = ?",
                ("projecting", time.time(), event_id),
            )

        self._in_tx(_tx)

    def mark_projected(
        self,
        *,
        event_id: str,
        event_seq: int,
        node_id: Optional[str],
        node_kind: Optional[str],
    ) -> None:
        now = time.time()

        def _tx(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE projection_state SET status = ?, event_seq = ?, node_id = ?, node_kind = ?, "
                "error = NULL, updated_ts = ? WHERE subject_id = ?",
                ("projected", int(event_seq), node_id, node_kind, now, event_id),
            )
            runtime_state = self.get_runtime_state(conn=conn)
            last_projected_seq = max(int(runtime_state.get("last_projected_seq", 0)), int(event_seq))
            runtime_state["last_projected_seq"] = last_projected_seq
            conn.execute(
                "UPDATE projection_state SET payload = ?, updated_ts = ? WHERE subject_id = ?",
                (dumps(runtime_state), now, RUNTIME_SYSTEM_SUBJECT_ID),
            )

        self._in_tx(_tx)

    def mark_projection_failed(self, *, event_id: str, error: str) -> None:
        def _tx(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE projection_state SET status = ?, error = ?, updated_ts = ? WHERE subject_id = ?",
                ("failed", error, time.time(), event_id),
            )

        self._in_tx(_tx)

    def update_runtime_state(self, **kwargs: Any) -> Dict[str, Any]:
        now = time.time()

        def _tx(conn: sqlite3.Connection) -> Dict[str, Any]:
            payload = self.get_runtime_state(conn=conn)
            payload.update(kwargs)
            conn.execute(
                "UPDATE projection_state SET payload = ?, updated_ts = ? WHERE subject_id = ?",
                (dumps(payload), now, RUNTIME_SYSTEM_SUBJECT_ID),
            )
            return payload

        return self._in_tx(_tx)

    def get_runtime_state(self, *, conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
        if conn is None:
            opened_conn = self._connect()
        else:
            opened_conn = conn
        try:
            row = opened_conn.execute(
                "SELECT payload FROM projection_state WHERE subject_id = ?",
                (RUNTIME_SYSTEM_SUBJECT_ID,),
            ).fetchone()
        finally:
            if conn is None:
                opened_conn.close()
        if row is None:
            return {"last_projected_seq": 0, "last_fade_ts": None, "last_evolve_ts": None}
        return loads(row["payload"])

    def claim_jobs(
        self,
        *,
        worker_id: str,
        limit: int,
        lease_seconds: float,
    ) -> List[StoredJob]:
        now = time.time()
        lease_until = now + float(lease_seconds)

        def _tx(conn: sqlite3.Connection) -> List[StoredJob]:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE "
                "((status IN ('queued', 'retry') AND available_at <= ?) "
                "OR (status = 'leased' AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?)) "
                "ORDER BY available_at ASC, created_ts ASC LIMIT ?",
                (now, now, int(limit)),
            ).fetchall()
            jobs: List[StoredJob] = []
            for row in rows:
                conn.execute(
                    "UPDATE jobs SET status = ?, lease_owner = ?, lease_expires_at = ?, updated_ts = ?, last_error = NULL "
                    "WHERE job_id = ?",
                    ("leased", worker_id, lease_until, now, row["job_id"]),
                )
                claimed = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (row["job_id"],)).fetchone()
                if claimed is not None:
                    jobs.append(self._row_to_job(claimed))
            return jobs

        return self._in_tx(_tx)

    def complete_job(self, job_id: str) -> None:
        now = time.time()

        def _tx(conn: sqlite3.Connection) -> None:
            row = conn.execute("SELECT attempts FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            attempts = int(row["attempts"]) if row is not None else 0
            conn.execute(
                "UPDATE jobs SET status = ?, attempts = ?, lease_owner = NULL, lease_expires_at = NULL, updated_ts = ? "
                "WHERE job_id = ?",
                ("succeeded", attempts + 1, now, job_id),
            )
            conn.execute(
                "INSERT INTO job_attempts(job_id, attempt_no, status, error, ts) VALUES (?, ?, ?, ?, ?)",
                (job_id, attempts + 1, "succeeded", None, now),
            )

        self._in_tx(_tx)

    def fail_job(self, *, job_id: str, error: str, retry_delay_s: float = 0.5) -> StoredJob:
        now = time.time()

        def _tx(conn: sqlite3.Connection) -> StoredJob:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise RuntimeError(f"Unknown job {job_id}")
            attempts = int(row["attempts"]) + 1
            max_attempts = int(row["max_attempts"])
            if attempts >= max_attempts:
                status = "failed"
                available_at = float(row["available_at"])
            else:
                status = "retry"
                available_at = now + max(0.0, float(retry_delay_s))
            conn.execute(
                "UPDATE jobs SET status = ?, attempts = ?, available_at = ?, lease_owner = NULL, "
                "lease_expires_at = NULL, updated_ts = ?, last_error = ? WHERE job_id = ?",
                (status, attempts, available_at, now, error, job_id),
            )
            conn.execute(
                "INSERT INTO job_attempts(job_id, attempt_no, status, error, ts) VALUES (?, ?, ?, ?, ?)",
                (job_id, attempts, status, error, now),
            )
            refreshed = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if refreshed is None:
                raise RuntimeError(f"Failed to update job {job_id}")
            return self._row_to_job(refreshed)

        return self._in_tx(_tx)

    def get_job(self, job_id: str) -> Optional[StoredJob]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        finally:
            conn.close()
        return None if row is None else self._row_to_job(row)

    def get_job_by_event(self, event_id: str) -> Optional[StoredJob]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM jobs WHERE event_id = ? ORDER BY created_ts ASC",
                (event_id,),
            ).fetchone()
        finally:
            conn.close()
        return None if row is None else self._row_to_job(row)

    def queue_depth(self) -> int:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM jobs WHERE status IN ('queued', 'retry', 'leased')",
            ).fetchone()
        finally:
            conn.close()
        return 0 if row is None else int(row["count"])

    def oldest_pending_age(self) -> Optional[float]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT created_ts FROM jobs WHERE status IN ('queued', 'retry', 'leased') ORDER BY created_ts ASC LIMIT 1",
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return max(0.0, time.time() - float(row["created_ts"]))

    def search_overlay_events(
        self,
        *,
        query_text: str,
        limit: int,
        session_id: Optional[str],
    ) -> List[OverlayHit]:
        lowered_query = query_text.strip()
        if not lowered_query:
            return []
        conn = self._connect()
        hits: List[OverlayHit] = []
        try:
            rows: Sequence[sqlite3.Row] = []
            try:
                rows = conn.execute(
                    "SELECT e.event_id, e.session_id, e.ts, e.payload, p.status, bm25(event_fts) AS rank "
                    "FROM event_fts "
                    "JOIN events e ON e.event_id = event_fts.event_id "
                    "JOIN projection_state p ON p.subject_id = e.event_id "
                    "WHERE event_fts MATCH ? AND p.status IN ('accepted', 'projecting', 'failed') "
                    "ORDER BY rank ASC LIMIT ?",
                    (lowered_query, int(limit * 4)),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
            for row in rows:
                payload = loads(row["payload"])
                snippet = str(payload.get("search_text") or payload.get("user_message") or "")[:240]
                score = self._overlay_score(
                    rank=float(row["rank"]),
                    ts=float(row["ts"]),
                    query_session_id=session_id,
                    event_session_id=str(row["session_id"]),
                    status=str(row["status"]),
                )
                hits.append(
                    OverlayHit(
                        event_id=str(row["event_id"]),
                        session_id=str(row["session_id"]),
                        snippet=snippet,
                        score=score,
                        ts=float(row["ts"]),
                        projection_status=str(row["status"]),
                    )
                )
            if hits:
                hits.sort(key=lambda item: item.score, reverse=True)
                return hits[:limit]

            rows = conn.execute(
                "SELECT e.event_id, e.session_id, e.ts, e.payload, p.status "
                "FROM events e JOIN projection_state p ON p.subject_id = e.event_id "
                "WHERE p.status IN ('accepted', 'projecting', 'failed') "
                "ORDER BY e.ts DESC LIMIT ?",
                (int(limit * 8),),
            ).fetchall()
            terms = {lowered_query.lower()}
            terms.update(part.lower() for part in lowered_query.split() if part.strip())
            for row in rows:
                payload = loads(row["payload"])
                text = str(payload.get("search_text") or payload.get("user_message") or "")
                text_lower = text.lower()
                match_count = sum(1 for term in terms if term and term in text_lower)
                if terms and match_count == 0:
                    continue
                score = self._overlay_score(
                    rank=float(-match_count),
                    ts=float(row["ts"]),
                    query_session_id=session_id,
                    event_session_id=str(row["session_id"]),
                    status=str(row["status"]),
                )
                hits.append(
                    OverlayHit(
                        event_id=str(row["event_id"]),
                        session_id=str(row["session_id"]),
                        snippet=text[:240],
                        score=score,
                        ts=float(row["ts"]),
                        projection_status=str(row["status"]),
                    )
                )
        finally:
            conn.close()
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:limit]

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "SQLiteRuntimeStore":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    @staticmethod
    def _overlay_score(
        *,
        rank: float,
        ts: float,
        query_session_id: Optional[str],
        event_session_id: str,
        status: str,
    ) -> float:
        now = time.time()
        age = max(1.0, now - ts)
        recency = 1.0 / (1.0 + (age / 300.0))
        text_score = 1.0 / (1.0 + max(0.0, rank))
        session_boost = 0.10 if query_session_id and query_session_id == event_session_id else 0.0
        pending_bonus = {"accepted": 0.25, "projecting": 0.18, "failed": 0.12}.get(status, 0.0)
        return min(1.5, text_score + 0.35 * recency + session_boost + pending_bonus)

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> StoredEvent:
        return StoredEvent(
            seq=int(row["seq"]),
            event_id=str(row["event_id"]),
            event_type=str(row["event_type"]),
            session_id=str(row["session_id"]),
            ts=float(row["ts"]),
            payload=loads(row["payload"]),
        )

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> StoredJob:
        return StoredJob(
            job_id=str(row["job_id"]),
            job_type=str(row["job_type"]),
            event_id=str(row["event_id"]) if row["event_id"] is not None else None,
            payload=loads(row["payload"]),
            status=str(row["status"]),
            attempts=int(row["attempts"]),
            max_attempts=int(row["max_attempts"]),
            available_at=float(row["available_at"]),
            lease_owner=str(row["lease_owner"]) if row["lease_owner"] is not None else None,
            lease_expires_at=(
                float(row["lease_expires_at"]) if row["lease_expires_at"] is not None else None
            ),
            dedupe_key=str(row["dedupe_key"]) if row["dedupe_key"] is not None else None,
            created_ts=float(row["created_ts"]),
            updated_ts=float(row["updated_ts"]),
            last_error=str(row["last_error"]) if row["last_error"] is not None else None,
        )

    @staticmethod
    def _row_to_projection(row: sqlite3.Row) -> ProjectionStatus:
        return ProjectionStatus(
            subject_id=str(row["subject_id"]),
            subject_type=str(row["subject_type"]),
            status=str(row["status"]),
            event_seq=int(row["event_seq"]) if row["event_seq"] is not None else None,
            node_id=str(row["node_id"]) if row["node_id"] is not None else None,
            node_kind=str(row["node_kind"]) if row["node_kind"] is not None else None,
            error=str(row["error"]) if row["error"] is not None else None,
            payload=loads(row["payload"]),
            updated_ts=float(row["updated_ts"]),
        )
