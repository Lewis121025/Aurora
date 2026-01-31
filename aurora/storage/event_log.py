from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

from aurora.utils.jsonx import dumps, loads


@dataclass
class Event:
    id: str
    ts: float
    user_id: str
    session_id: str
    type: str
    payload: Dict[str, Any]


class SQLiteEventLog:
    """Append-only event log.

    Table schema:
      events(seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE, ts REAL, user_id TEXT, session_id TEXT, type TEXT, payload TEXT)
    """

    def __init__(self, path: str):
        self.path = path
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS events ("
            "seq INTEGER PRIMARY KEY AUTOINCREMENT,"
            "id TEXT UNIQUE,"
            "ts REAL,"
            "user_id TEXT,"
            "session_id TEXT,"
            "type TEXT,"
            "payload TEXT"
            ")"
        )
        self._conn.commit()

    def append(self, event: Event) -> int:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO events(id, ts, user_id, session_id, type, payload) VALUES (?, ?, ?, ?, ?, ?)",
            (event.id, event.ts, event.user_id, event.session_id, event.type, dumps(event.payload)),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def get_seq_by_id(self, event_id: str) -> Optional[int]:
        cur = self._conn.cursor()
        cur.execute("SELECT seq FROM events WHERE id = ?", (event_id,))
        row = cur.fetchone()
        return int(row[0]) if row else None

    def iter_events(self, *, after_seq: int = 0, user_id: Optional[str] = None) -> Iterable[Tuple[int, Event]]:
        cur = self._conn.cursor()
        if user_id:
            cur.execute(
                "SELECT seq, id, ts, user_id, session_id, type, payload FROM events WHERE seq > ? AND user_id = ? ORDER BY seq ASC",
                (after_seq, user_id),
            )
        else:
            cur.execute(
                "SELECT seq, id, ts, user_id, session_id, type, payload FROM events WHERE seq > ? ORDER BY seq ASC",
                (after_seq,),
            )
        for seq, _id, ts, uid, sid, typ, payload in cur.fetchall():
            yield int(seq), Event(
                id=str(_id),
                ts=float(ts),
                user_id=str(uid),
                session_id=str(sid),
                type=str(typ),
                payload=loads(payload),
            )

    def close(self) -> None:
        """Explicitly close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "SQLiteEventLog":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures connection is closed."""
        self.close()
