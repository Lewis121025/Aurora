"""Snapshot store and short-lived session transcript persistence."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from aurora.field_engine import MemoryKernel, kernel_from_snapshot, snapshot_from_kernel


@dataclass(frozen=True)
class SessionTurn:
    session_id: str
    ordinal: int
    role: str
    text: str
    created_at: float
    event_id: str


class SQLiteSnapshotStore:
    """Persist the evolving kernel as compressed snapshots plus operation logs."""

    def __init__(self, db_path: str | Path, *, max_snapshots: int = 256):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_snapshots = max_snapshots
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;

                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    step INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    payload_blob BLOB NOT NULL,
                    payload_size INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_snapshots_created_at ON snapshots(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_snapshots_step ON snapshots(step DESC);

                CREATE TABLE IF NOT EXISTS operations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    step INTEGER NOT NULL,
                    op_kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_operations_created_at ON operations(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_operations_kind ON operations(op_kind);

                CREATE TABLE IF NOT EXISTS session_turns (
                    session_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    text TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    event_id TEXT NOT NULL,
                    PRIMARY KEY(session_id, ordinal)
                );
                CREATE INDEX IF NOT EXISTS idx_session_turns_created ON session_turns(session_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )

    def load_latest_kernel(self) -> Optional[MemoryKernel]:
        with self._lock:
            row = self.conn.execute("SELECT payload_blob FROM snapshots ORDER BY id DESC LIMIT 1").fetchone()
        if row is None:
            return None
        payload = json.loads(zlib.decompress(row["payload_blob"]).decode("utf-8"))
        return kernel_from_snapshot(payload)

    def save_snapshot(
        self,
        kernel: MemoryKernel,
        *,
        reason: str,
        operation_summary: Optional[Dict[str, Any]] = None,
    ) -> int:
        payload_json = json.dumps(snapshot_from_kernel(kernel), ensure_ascii=False, separators=(",", ":"))
        payload_blob = sqlite3.Binary(zlib.compress(payload_json.encode("utf-8"), level=6))
        created_at = time.time()
        with self._lock, self.conn:
            cursor = self.conn.execute(
                """
                INSERT INTO snapshots (created_at, step, reason, payload_blob, payload_size)
                VALUES (?, ?, ?, ?, ?)
                """,
                (created_at, kernel.step, reason, payload_blob, len(payload_json)),
            )
            if operation_summary is not None:
                self.conn.execute(
                    """
                    INSERT INTO operations (created_at, step, op_kind, payload_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        created_at,
                        kernel.step,
                        reason,
                        json.dumps(operation_summary, ensure_ascii=False, separators=(",", ":")),
                    ),
                )
            self.conn.execute(
                """
                INSERT INTO meta (key, value) VALUES ('updated_at', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(created_at),),
            )
            self._prune_snapshots_locked()
            if cursor.lastrowid is None:
                raise RuntimeError("snapshot insert did not return lastrowid")
            return int(cursor.lastrowid)

    def log_operation(self, step: int, op_kind: str, payload: Dict[str, Any]) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO operations (created_at, step, op_kind, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    time.time(),
                    step,
                    op_kind,
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                ),
            )

    def list_operations(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT created_at, step, op_kind, payload_json FROM operations ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "created_at": float(row["created_at"]),
                "step": int(row["step"]),
                "op_kind": str(row["op_kind"]),
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    def append_session_turn(
        self,
        session_id: str,
        role: str,
        text: str,
        *,
        event_id: str,
        created_at: float | None = None,
    ) -> SessionTurn:
        timestamp = time.time() if created_at is None else created_at
        with self._lock, self.conn:
            row = self.conn.execute(
                "SELECT COALESCE(MAX(ordinal), 0) + 1 AS next_ordinal FROM session_turns WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            ordinal = int(row["next_ordinal"]) if row is not None else 1
            self.conn.execute(
                """
                INSERT INTO session_turns (session_id, ordinal, role, text, created_at, event_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, ordinal, role, text, timestamp, event_id),
            )
        return SessionTurn(
            session_id=session_id,
            ordinal=ordinal,
            role=role,
            text=text,
            created_at=timestamp,
            event_id=event_id,
        )

    def list_session_turns(self, session_id: str, *, limit: int | None = None) -> List[SessionTurn]:
        sql = """
            SELECT session_id, ordinal, role, text, created_at, event_id
            FROM session_turns
            WHERE session_id = ?
            ORDER BY ordinal DESC
        """
        params: list[Any] = [session_id]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self._lock:
            rows = self.conn.execute(sql, tuple(params)).fetchall()
        turns = [
            SessionTurn(
                session_id=str(row["session_id"]),
                ordinal=int(row["ordinal"]),
                role=str(row["role"]),
                text=str(row["text"]),
                created_at=float(row["created_at"]),
                event_id=str(row["event_id"]),
            )
            for row in rows
        ]
        turns.reverse()
        return turns

    def session_count(self) -> int:
        with self._lock:
            row = self.conn.execute("SELECT COUNT(DISTINCT session_id) AS n FROM session_turns").fetchone()
        return int(row["n"]) if row is not None else 0

    def snapshot_count(self) -> int:
        with self._lock:
            row = self.conn.execute("SELECT COUNT(*) AS n FROM snapshots").fetchone()
        return int(row["n"]) if row is not None else 0

    def latest_snapshot_meta(self) -> Dict[str, Any]:
        with self._lock:
            row = self.conn.execute(
                "SELECT id, created_at, step, reason, payload_size FROM snapshots ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return {}
        return {
            "snapshot_id": int(row["id"]),
            "created_at": float(row["created_at"]),
            "step": int(row["step"]),
            "reason": str(row["reason"]),
            "payload_size": int(row["payload_size"]),
        }

    def export_latest_json(self, path: str | Path) -> Path:
        target = Path(path)
        with self._lock:
            row = self.conn.execute("SELECT payload_blob FROM snapshots ORDER BY id DESC LIMIT 1").fetchone()
        if row is None:
            raise FileNotFoundError("no snapshot in store")
        payload = zlib.decompress(row["payload_blob"]).decode("utf-8")
        target.write_text(json.dumps(json.loads(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def _prune_snapshots_locked(self) -> None:
        if self.max_snapshots <= 0:
            return
        rows = self.conn.execute(
            "SELECT id FROM snapshots ORDER BY id DESC LIMIT -1 OFFSET ?",
            (self.max_snapshots,),
        ).fetchall()
        if not rows:
            return
        snapshot_ids = [int(row["id"]) for row in rows]
        placeholders = ",".join("?" for _ in snapshot_ids)
        self.conn.execute(f"DELETE FROM snapshots WHERE id IN ({placeholders})", snapshot_ids)


__all__ = ["SQLiteSnapshotStore", "SessionTurn"]
