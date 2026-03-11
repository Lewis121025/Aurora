from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar, cast

import numpy as np

from aurora.utils.jsonx import dumps, loads

RUNTIME_SYSTEM_SUBJECT_ID = "__runtime__"
TxResult = TypeVar("TxResult")


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

    def __init__(
        self,
        path: str,
        *,
        runtime_schema_version: str,
        memory_state_version: str,
    ):
        self.path = path
        self.runtime_schema_version = runtime_schema_version
        self.memory_state_version = memory_state_version
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
            existing_tables = {
                str(row["name"])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
            }
            if "schema_meta" not in existing_tables and existing_tables & {
                "events",
                "jobs",
                "projection_state",
            }:
                raise ValueError(
                    "Detected legacy Aurora runtime DB. Aurora V7 requires a fresh data directory."
                )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_meta ("
                "key TEXT PRIMARY KEY,"
                "value TEXT NOT NULL"
                ")"
            )
            runtime_row = conn.execute(
                "SELECT value FROM schema_meta WHERE key = 'runtime_schema_version'"
            ).fetchone()
            memory_row = conn.execute(
                "SELECT value FROM schema_meta WHERE key = 'memory_state_version'"
            ).fetchone()
            if runtime_row is not None and str(runtime_row["value"]) != self.runtime_schema_version:
                raise ValueError(
                    "Runtime DB schema version mismatch. Aurora V7 requires a fresh data directory."
                )
            if memory_row is not None and str(memory_row["value"]) != self.memory_state_version:
                raise ValueError(
                    "Derived memory schema mismatch. Aurora V7 requires a fresh data directory."
                )
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
                "CREATE TABLE IF NOT EXISTS derived_nodes ("
                "node_id TEXT PRIMARY KEY,"
                "kind TEXT NOT NULL,"
                "state TEXT NOT NULL"
                ")"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS derived_vectors ("
                "node_id TEXT PRIMARY KEY,"
                "kind TEXT NOT NULL,"
                "dim INTEGER NOT NULL,"
                "data BLOB NOT NULL,"
                "FOREIGN KEY(node_id) REFERENCES derived_nodes(node_id) ON DELETE CASCADE"
                ")"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS derived_edges ("
                "src TEXT NOT NULL,"
                "dst TEXT NOT NULL,"
                "edge_type TEXT NOT NULL,"
                "sign INTEGER NOT NULL,"
                "weight REAL NOT NULL,"
                "confidence REAL NOT NULL,"
                "provenance TEXT NOT NULL,"
                "PRIMARY KEY(src, dst)"
                ")"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS derived_identity_state ("
                "subject_id TEXT PRIMARY KEY,"
                "state TEXT NOT NULL"
                ")"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS derived_schema_state ("
                "subject_id TEXT PRIMARY KEY,"
                "state TEXT NOT NULL"
                ")"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS derived_modes ("
                "mode_id TEXT PRIMARY KEY,"
                "state TEXT NOT NULL"
                ")"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS derived_metric_state ("
                "subject_id TEXT PRIMARY KEY,"
                "dim INTEGER NOT NULL,"
                "rank INTEGER NOT NULL,"
                "seed INTEGER NOT NULL,"
                "L BLOB NOT NULL,"
                "G BLOB NOT NULL,"
                "t INTEGER NOT NULL"
                ")"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS derived_kde_state ("
                "subject_id TEXT PRIMARY KEY,"
                "dim INTEGER NOT NULL,"
                "reservoir INTEGER NOT NULL,"
                "k_sigma INTEGER NOT NULL,"
                "seed INTEGER NOT NULL,"
                "vec_count INTEGER NOT NULL,"
                "vecs BLOB NOT NULL"
                ")"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS derived_runtime_state ("
                "subject_id TEXT PRIMARY KEY,"
                "step INTEGER NOT NULL,"
                "recent_semantic_texts TEXT NOT NULL,"
                "anchors TEXT NOT NULL,"
                "core_anchor_ids TEXT NOT NULL,"
                "view_stats TEXT NOT NULL,"
                "graph_metrics TEXT NOT NULL,"
                "consolidator TEXT NOT NULL"
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
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
                ("runtime_schema_version", self.runtime_schema_version),
            )
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
                ("memory_state_version", self.memory_state_version),
            )
        finally:
            conn.close()

    def _in_tx(self, fn: Callable[[sqlite3.Connection], TxResult]) -> TxResult:
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
        return self._loads_object(row["payload"])

    def apply_projection_delta(
        self,
        *,
        mem: Any,
        delta: Any,
        edge_upserts: Sequence[Tuple[str, str, Any]],
        edge_deletes: Sequence[Tuple[str, str]],
        last_projected_seq: int,
        runtime_state: Dict[str, Any],
        event_id: Optional[str] = None,
        event_seq: Optional[int] = None,
        node_id: Optional[str] = None,
        node_kind: Optional[str] = None,
    ) -> None:
        now = time.time()

        def _node_snapshot(kind: str, node_id_value: str) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]:
            if kind == "plot":
                plot = mem.plots.get(node_id_value)
                if plot is None:
                    return None, None
                state = plot.to_state_dict()
                state.pop("messages", None)
                state.pop("embedding", None)
                return state, plot.embedding
            if kind == "summary":
                summary = mem.summaries.get(node_id_value)
                if summary is None:
                    return None, None
                state = summary.to_state_dict()
                state.pop("embedding", None)
                return state, summary.embedding
            if kind == "story":
                story = mem.stories.get(node_id_value)
                if story is None:
                    return None, None
                state = story.to_state_dict()
                state.pop("centroid", None)
                return state, story.centroid
            if kind == "theme":
                theme = mem.themes.get(node_id_value)
                if theme is None:
                    return None, None
                state = theme.to_state_dict()
                state.pop("prototype", None)
                return state, theme.prototype
            if kind == "anchor":
                payload = mem.anchor_nodes.get(node_id_value)
                if payload is None:
                    return None, None
                return (
                    {
                        "id": str(payload.get("id", node_id_value)),
                        "label": str(payload.get("label", "")),
                        "pinned": bool(payload.get("pinned", True)),
                    },
                    np.asarray(payload.get("embedding"), dtype=np.float32),
                )
            raise ValueError(f"Unsupported delta node kind: {kind}")

        def _upsert_node(conn: sqlite3.Connection, *, kind: str, node_id_value: str) -> None:
            state, vector = _node_snapshot(kind, node_id_value)
            if state is None:
                return
            conn.execute(
                "INSERT INTO derived_nodes(node_id, kind, state) VALUES (?, ?, ?) "
                "ON CONFLICT(node_id) DO UPDATE SET kind = excluded.kind, state = excluded.state",
                (node_id_value, kind, dumps(state)),
            )
            if vector is not None and vector.size > 0:
                conn.execute(
                    "INSERT INTO derived_vectors(node_id, kind, dim, data) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(node_id) DO UPDATE SET kind = excluded.kind, dim = excluded.dim, data = excluded.data",
                    (
                        node_id_value,
                        kind,
                        int(vector.shape[0]),
                        self._vector_to_blob(vector),
                    ),
                )
            else:
                conn.execute("DELETE FROM derived_vectors WHERE node_id = ?", (node_id_value,))

        def _anchor_payloads() -> Dict[str, Dict[str, Any]]:
            return {
                anchor_id: {
                    "id": str(payload.get("id", anchor_id)),
                    "label": str(payload.get("label", "")),
                    "pinned": bool(payload.get("pinned", True)),
                }
                for anchor_id, payload in mem.anchor_nodes.items()
            }

        def _upsert_modes(conn: sqlite3.Connection) -> None:
            conn.execute("DELETE FROM derived_modes")
            for mode in mem.modes.values():
                conn.execute(
                    "INSERT INTO derived_modes(mode_id, state) VALUES (?, ?)",
                    (mode.id, dumps(mode.to_state_dict())),
                )

        def _upsert_metric(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO derived_metric_state(subject_id, dim, rank, seed, L, G, t) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(subject_id) DO UPDATE SET dim = excluded.dim, rank = excluded.rank, "
                "seed = excluded.seed, L = excluded.L, G = excluded.G, t = excluded.t",
                (
                    "metric",
                    int(mem.metric.dim),
                    int(mem.metric.rank),
                    int(getattr(mem.metric, "_seed", 0)),
                    self._matrix_to_blob(mem.metric.L),
                    self._matrix_to_blob(mem.metric.G),
                    int(mem.metric.t),
                ),
            )

        def _upsert_kde(conn: sqlite3.Connection) -> None:
            kde_matrix = (
                np.vstack(mem.kde._vecs).astype(np.float32)
                if getattr(mem.kde, "_vecs", None)
                else np.zeros((0, mem.kde.dim), dtype=np.float32)
            )
            conn.execute(
                "INSERT INTO derived_kde_state(subject_id, dim, reservoir, k_sigma, seed, vec_count, vecs) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(subject_id) DO UPDATE SET dim = excluded.dim, reservoir = excluded.reservoir, "
                "k_sigma = excluded.k_sigma, seed = excluded.seed, vec_count = excluded.vec_count, vecs = excluded.vecs",
                (
                    "kde",
                    int(mem.kde.dim),
                    int(mem.kde.reservoir),
                    int(mem.kde.k_sigma),
                    int(getattr(mem.kde, "_seed", 0)),
                    int(kde_matrix.shape[0]),
                    self._matrix_to_blob(kde_matrix),
                ),
            )

        def _upsert_runtime(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO derived_runtime_state("
                "subject_id, step, recent_semantic_texts, anchors, core_anchor_ids, view_stats, graph_metrics, consolidator"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(subject_id) DO UPDATE SET step = excluded.step, recent_semantic_texts = excluded.recent_semantic_texts, "
                "anchors = excluded.anchors, core_anchor_ids = excluded.core_anchor_ids, view_stats = excluded.view_stats, "
                "graph_metrics = excluded.graph_metrics, consolidator = excluded.consolidator",
                (
                    "runtime",
                    int(mem.step),
                    dumps(list(mem.recent_semantic_texts)),
                    dumps(_anchor_payloads()),
                    dumps(list(mem.core_anchor_ids)),
                    dumps(mem.view_stats.to_state_dict()),
                    dumps(dict(mem.graph_metrics)),
                    dumps(mem.consolidator.to_state_dict()),
                ),
            )

        def _tx(conn: sqlite3.Connection) -> None:
            for kind, node_ids in getattr(delta, "delete_nodes", {}).items():
                for node_id_value in node_ids:
                    conn.execute("DELETE FROM derived_vectors WHERE node_id = ?", (node_id_value,))
                    conn.execute("DELETE FROM derived_nodes WHERE node_id = ?", (node_id_value,))

            for src, dst in edge_deletes:
                conn.execute("DELETE FROM derived_edges WHERE src = ? AND dst = ?", (src, dst))

            for kind, node_ids in getattr(delta, "upsert_nodes", {}).items():
                for node_id_value in node_ids:
                    _upsert_node(conn, kind=kind, node_id_value=node_id_value)

            for src, dst, belief in edge_upserts:
                conn.execute(
                    "INSERT INTO derived_edges(src, dst, edge_type, sign, weight, confidence, provenance) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(src, dst) DO UPDATE SET edge_type = excluded.edge_type, sign = excluded.sign, "
                    "weight = excluded.weight, confidence = excluded.confidence, provenance = excluded.provenance",
                    (
                        src,
                        dst,
                        belief.edge_type,
                        int(belief.sign),
                        float(belief.weight),
                        float(belief.confidence),
                        belief.provenance,
                    ),
                )

            if getattr(delta, "persist_identity", False):
                conn.execute(
                    "INSERT INTO derived_identity_state(subject_id, state) VALUES (?, ?) "
                    "ON CONFLICT(subject_id) DO UPDATE SET state = excluded.state",
                    ("identity", dumps(mem.identity.to_state_dict())),
                )
            if getattr(delta, "persist_schema", False):
                conn.execute(
                    "INSERT INTO derived_schema_state(subject_id, state) VALUES (?, ?) "
                    "ON CONFLICT(subject_id) DO UPDATE SET state = excluded.state",
                    ("schema", dumps(mem.schema.to_state_dict())),
                )
            if getattr(delta, "persist_modes", False):
                _upsert_modes(conn)
            if getattr(delta, "persist_metric", False):
                _upsert_metric(conn)
            if getattr(delta, "persist_kde", False):
                _upsert_kde(conn)
            if getattr(delta, "persist_runtime_state", False):
                _upsert_runtime(conn)

            updated_runtime_state = dict(runtime_state)
            updated_runtime_state["last_projected_seq"] = int(last_projected_seq)
            conn.execute(
                "UPDATE projection_state SET payload = ?, updated_ts = ? WHERE subject_id = ?",
                (dumps(updated_runtime_state), now, RUNTIME_SYSTEM_SUBJECT_ID),
            )
            if event_id is not None:
                conn.execute(
                    "UPDATE projection_state SET status = ?, event_seq = ?, node_id = ?, node_kind = ?, "
                    "error = NULL, updated_ts = ? WHERE subject_id = ?",
                    ("projected", event_seq, node_id, node_kind, now, event_id),
                )

        self._in_tx(_tx)

    def replace_derived_state(
        self,
        *,
        mem: Any,
        last_projected_seq: int,
        runtime_state: Dict[str, Any],
        event_id: Optional[str] = None,
        event_seq: Optional[int] = None,
        node_id: Optional[str] = None,
        node_kind: Optional[str] = None,
    ) -> None:
        now = time.time()

        def _tx(conn: sqlite3.Connection) -> None:
            for table_name in (
                "derived_vectors",
                "derived_edges",
                "derived_nodes",
                "derived_modes",
                "derived_identity_state",
                "derived_schema_state",
                "derived_metric_state",
                "derived_kde_state",
                "derived_runtime_state",
            ):
                conn.execute(f"DELETE FROM {table_name}")

            def _insert_node(
                *,
                node_id: str,
                kind: str,
                state: Dict[str, Any],
                vector: Optional[np.ndarray],
            ) -> None:
                conn.execute(
                    "INSERT INTO derived_nodes(node_id, kind, state) VALUES (?, ?, ?)",
                    (node_id, kind, dumps(state)),
                )
                if vector is not None and vector.size > 0:
                    conn.execute(
                        "INSERT INTO derived_vectors(node_id, kind, dim, data) VALUES (?, ?, ?, ?)",
                        (
                            node_id,
                            kind,
                            int(vector.shape[0]),
                            self._vector_to_blob(vector),
                        ),
                    )

            for plot in mem.plots.values():
                state = plot.to_state_dict()
                state.pop("messages", None)
                state.pop("embedding", None)
                _insert_node(node_id=plot.id, kind="plot", state=state, vector=plot.embedding)

            for summary in mem.summaries.values():
                state = summary.to_state_dict()
                state.pop("embedding", None)
                _insert_node(
                    node_id=summary.id,
                    kind="summary",
                    state=state,
                    vector=summary.embedding,
                )

            for story in mem.stories.values():
                state = story.to_state_dict()
                state.pop("centroid", None)
                vector = story.centroid if story.centroid is not None else None
                _insert_node(node_id=story.id, kind="story", state=state, vector=vector)

            for theme in mem.themes.values():
                state = theme.to_state_dict()
                state.pop("prototype", None)
                vector = theme.prototype if theme.prototype is not None else None
                _insert_node(node_id=theme.id, kind="theme", state=state, vector=vector)

            anchor_payloads: Dict[str, Dict[str, Any]] = {}
            for anchor_id, payload in mem.anchor_nodes.items():
                anchor_payloads[anchor_id] = {
                    "id": str(payload.get("id", anchor_id)),
                    "label": str(payload.get("label", "")),
                    "pinned": bool(payload.get("pinned", True)),
                }
                vector = np.asarray(payload.get("embedding"), dtype=np.float32)
                _insert_node(
                    node_id=anchor_id,
                    kind="anchor",
                    state=anchor_payloads[anchor_id],
                    vector=vector,
                )

            for src, dst, belief in mem.graph.iter_edge_items():
                conn.execute(
                    "INSERT INTO derived_edges(src, dst, edge_type, sign, weight, confidence, provenance) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        src,
                        dst,
                        belief.edge_type,
                        int(belief.sign),
                        float(belief.weight),
                        float(belief.confidence),
                        belief.provenance,
                    ),
                )

            conn.execute(
                "INSERT INTO derived_identity_state(subject_id, state) VALUES (?, ?)",
                ("identity", dumps(mem.identity.to_state_dict())),
            )
            conn.execute(
                "INSERT INTO derived_schema_state(subject_id, state) VALUES (?, ?)",
                ("schema", dumps(mem.schema.to_state_dict())),
            )
            for mode in mem.modes.values():
                conn.execute(
                    "INSERT INTO derived_modes(mode_id, state) VALUES (?, ?)",
                    (mode.id, dumps(mode.to_state_dict())),
                )

            conn.execute(
                "INSERT INTO derived_metric_state(subject_id, dim, rank, seed, L, G, t) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "metric",
                    int(mem.metric.dim),
                    int(mem.metric.rank),
                    int(getattr(mem.metric, "_seed", 0)),
                    self._matrix_to_blob(mem.metric.L),
                    self._matrix_to_blob(mem.metric.G),
                    int(mem.metric.t),
                ),
            )

            kde_matrix = (
                np.vstack(mem.kde._vecs).astype(np.float32) if getattr(mem.kde, "_vecs", None) else np.zeros((0, mem.kde.dim), dtype=np.float32)
            )
            conn.execute(
                "INSERT INTO derived_kde_state(subject_id, dim, reservoir, k_sigma, seed, vec_count, vecs) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "kde",
                    int(mem.kde.dim),
                    int(mem.kde.reservoir),
                    int(mem.kde.k_sigma),
                    int(getattr(mem.kde, "_seed", 0)),
                    int(kde_matrix.shape[0]),
                    self._matrix_to_blob(kde_matrix),
                ),
            )

            conn.execute(
                "INSERT INTO derived_runtime_state("
                "subject_id, step, recent_semantic_texts, anchors, core_anchor_ids, view_stats, graph_metrics, consolidator"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "runtime",
                    int(mem.step),
                    dumps(list(mem.recent_semantic_texts)),
                    dumps(anchor_payloads),
                    dumps(list(mem.core_anchor_ids)),
                    dumps(mem.view_stats.to_state_dict()),
                    dumps(dict(mem.graph_metrics)),
                    dumps(mem.consolidator.to_state_dict()),
                ),
            )

            updated_runtime_state = dict(runtime_state)
            updated_runtime_state["last_projected_seq"] = int(last_projected_seq)
            conn.execute(
                "UPDATE projection_state SET payload = ?, updated_ts = ? WHERE subject_id = ?",
                (dumps(updated_runtime_state), now, RUNTIME_SYSTEM_SUBJECT_ID),
            )
            if event_id is not None:
                conn.execute(
                    "UPDATE projection_state SET status = ?, event_seq = ?, node_id = ?, node_kind = ?, "
                    "error = NULL, updated_ts = ? WHERE subject_id = ?",
                    ("projected", event_seq, node_id, node_kind, now, event_id),
                )

        self._in_tx(_tx)

    def load_derived_state(self) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        try:
            runtime_row = conn.execute(
                "SELECT * FROM derived_runtime_state WHERE subject_id = 'runtime'"
            ).fetchone()
            if runtime_row is None:
                return None
            node_rows = conn.execute(
                "SELECT n.node_id, n.kind, n.state, v.dim, v.data "
                "FROM derived_nodes n LEFT JOIN derived_vectors v ON v.node_id = n.node_id"
            ).fetchall()
            edge_rows = conn.execute("SELECT * FROM derived_edges").fetchall()
            identity_row = conn.execute(
                "SELECT state FROM derived_identity_state WHERE subject_id = 'identity'"
            ).fetchone()
            schema_row = conn.execute(
                "SELECT state FROM derived_schema_state WHERE subject_id = 'schema'"
            ).fetchone()
            mode_rows = conn.execute("SELECT mode_id, state FROM derived_modes").fetchall()
            metric_row = conn.execute(
                "SELECT * FROM derived_metric_state WHERE subject_id = 'metric'"
            ).fetchone()
            kde_row = conn.execute(
                "SELECT * FROM derived_kde_state WHERE subject_id = 'kde'"
            ).fetchone()
        finally:
            conn.close()

        if identity_row is None or schema_row is None or metric_row is None or kde_row is None:
            return None

        plots: Dict[str, Dict[str, Any]] = {}
        summaries: Dict[str, Dict[str, Any]] = {}
        stories: Dict[str, Dict[str, Any]] = {}
        themes: Dict[str, Dict[str, Any]] = {}
        anchors: Dict[str, Dict[str, Any]] = {}
        vindex_items: List[Dict[str, Any]] = []

        for row in node_rows:
            node_id = str(row["node_id"])
            kind = str(row["kind"])
            state = self._loads_object(row["state"])
            vector = (
                self._vector_from_blob(cast(bytes, row["data"]), int(row["dim"]))
                if row["data"] is not None
                else None
            )
            if kind == "plot":
                if vector is not None:
                    state["embedding"] = vector.tolist()
                    vindex_items.append({"id": node_id, "kind": kind, "vec": vector.tolist()})
                plots[node_id] = state
            elif kind == "summary":
                if vector is not None:
                    state["embedding"] = vector.tolist()
                    vindex_items.append({"id": node_id, "kind": kind, "vec": vector.tolist()})
                summaries[node_id] = state
            elif kind == "story":
                if vector is not None:
                    state["centroid"] = vector.tolist()
                    vindex_items.append({"id": node_id, "kind": kind, "vec": vector.tolist()})
                stories[node_id] = state
            elif kind == "theme":
                if vector is not None:
                    state["prototype"] = vector.tolist()
                    vindex_items.append({"id": node_id, "kind": kind, "vec": vector.tolist()})
                themes[node_id] = state
            elif kind == "anchor":
                anchor_state = dict(state)
                if vector is not None:
                    anchor_state["embedding"] = vector.tolist()
                anchors[node_id] = anchor_state

        metric_L = self._matrix_from_blob(
            cast(bytes, metric_row["L"]),
            rows=int(metric_row["rank"]),
            cols=int(metric_row["dim"]),
        )
        metric_G = self._matrix_from_blob(
            cast(bytes, metric_row["G"]),
            rows=int(metric_row["rank"]),
            cols=int(metric_row["dim"]),
        )
        kde_matrix = self._matrix_from_blob(
            cast(bytes, kde_row["vecs"]),
            rows=int(kde_row["vec_count"]),
            cols=int(kde_row["dim"]),
        )

        return {
            "schema_version": self.memory_state_version,
            "kde": {
                "dim": int(kde_row["dim"]),
                "reservoir": int(kde_row["reservoir"]),
                "k_sigma": int(kde_row["k_sigma"]),
                "seed": int(kde_row["seed"]),
                "vecs": kde_matrix.tolist(),
            },
            "metric": {
                "dim": int(metric_row["dim"]),
                "rank": int(metric_row["rank"]),
                "seed": int(metric_row["seed"]),
                "L": metric_L.tolist(),
                "G": metric_G.tolist(),
                "t": int(metric_row["t"]),
            },
            "graph": {
                "edges": [
                    {
                        "src": str(row["src"]),
                        "dst": str(row["dst"]),
                        "belief": {
                            "edge_type": str(row["edge_type"]),
                            "sign": int(row["sign"]),
                            "weight": float(row["weight"]),
                            "confidence": float(row["confidence"]),
                            "provenance": str(row["provenance"]),
                        },
                    }
                    for row in edge_rows
                ],
                "edge_version": len(edge_rows),
            },
            "vindex": {
                "dim": int(metric_row["dim"]),
                "items": vindex_items,
            },
            "plots": plots,
            "summaries": summaries,
            "stories": stories,
            "themes": themes,
            "schema": self._loads_object(schema_row["state"]),
            "consolidator": self._loads_object(runtime_row["consolidator"]),
            "identity": self._loads_object(identity_row["state"]),
            "modes": {
                str(row["mode_id"]): self._loads_object(row["state"])
                for row in mode_rows
            },
            "recent_semantic_texts": cast(
                List[str], loads(str(runtime_row["recent_semantic_texts"]))
            ),
            "step": int(runtime_row["step"]),
            "anchors": cast(Dict[str, Any], loads(str(runtime_row["anchors"]))),
            "core_anchor_ids": cast(List[str], loads(str(runtime_row["core_anchor_ids"]))),
            "view_stats": self._loads_object(runtime_row["view_stats"]),
            "graph_metrics": self._loads_object(runtime_row["graph_metrics"]),
        }

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
                payload = self._loads_object(row["payload"])
                snippet = str(payload.get("search_text") or "")[:240]
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
                payload = self._loads_object(row["payload"])
                text = str(payload.get("search_text") or "")
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
    def _vector_to_blob(vec: np.ndarray) -> bytes:
        return np.asarray(vec, dtype="<f4").tobytes(order="C")

    @staticmethod
    def _vector_from_blob(blob: bytes, dim: int) -> np.ndarray:
        return np.frombuffer(blob, dtype="<f4", count=dim).astype(np.float32)

    @staticmethod
    def _matrix_to_blob(matrix: np.ndarray) -> bytes:
        return np.asarray(matrix, dtype="<f4").tobytes(order="C")

    @staticmethod
    def _matrix_from_blob(blob: bytes, *, rows: int, cols: int) -> np.ndarray:
        if rows == 0 or cols == 0:
            return np.zeros((rows, cols), dtype=np.float32)
        array = np.frombuffer(blob, dtype="<f4", count=rows * cols)
        return array.reshape((rows, cols)).astype(np.float32)

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> StoredEvent:
        return StoredEvent(
            seq=int(row["seq"]),
            event_id=str(row["event_id"]),
            event_type=str(row["event_type"]),
            session_id=str(row["session_id"]),
            ts=float(row["ts"]),
            payload=SQLiteRuntimeStore._loads_object(row["payload"]),
        )

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> StoredJob:
        return StoredJob(
            job_id=str(row["job_id"]),
            job_type=str(row["job_type"]),
            event_id=str(row["event_id"]) if row["event_id"] is not None else None,
            payload=SQLiteRuntimeStore._loads_object(row["payload"]),
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
            payload=SQLiteRuntimeStore._loads_object(row["payload"]),
            updated_ts=float(row["updated_ts"]),
        )

    @staticmethod
    def _loads_object(raw: str) -> Dict[str, Any]:
        payload = loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("Expected JSON object payload in runtime store")
        return cast(Dict[str, Any], payload)
