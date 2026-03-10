from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Dict, Iterable, Optional, Tuple

from aurora.utils.jsonx import dumps, loads


@dataclass
class Event:
    id: str
    ts: float
    session_id: str
    type: str
    payload: Dict[str, Any]


class SQLiteEventLog:
    """仅追加事件日志。

    表模式：
      events(seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE, ts REAL, session_id TEXT, type TEXT, payload TEXT)
    """

    def __init__(self, path: str):
        self.path = path
        self._conn: sqlite3.Connection | None = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS events ("
            "seq INTEGER PRIMARY KEY AUTOINCREMENT,"
            "id TEXT UNIQUE,"
            "ts REAL,"
            "session_id TEXT,"
            "type TEXT,"
            "payload TEXT"
            ")"
        )
        self._conn.commit()

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Event log connection is closed")
        return self._conn

    def append(self, event: Event) -> int:
        conn = self._require_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO events(id, ts, session_id, type, payload) VALUES (?, ?, ?, ?, ?)",
            (event.id, event.ts, event.session_id, event.type, dumps(event.payload)),
        )
        conn.commit()
        return int(cur.lastrowid or 0)

    def get_seq_by_id(self, event_id: str) -> Optional[int]:
        cur = self._require_conn().cursor()
        cur.execute("SELECT seq FROM events WHERE id = ?", (event_id,))
        row = cur.fetchone()
        return int(row[0]) if row else None

    def update_payload(self, event_id: str, payload: Dict[str, Any]) -> None:
        conn = self._require_conn()
        conn.execute(
            "UPDATE events SET payload = ? WHERE id = ?",
            (dumps(payload), event_id),
        )
        conn.commit()

    def iter_events(self, *, after_seq: int = 0) -> Iterable[Tuple[int, Event]]:
        cur = self._require_conn().cursor()
        cur.execute(
            "SELECT seq, id, ts, session_id, type, payload FROM events WHERE seq > ? ORDER BY seq ASC",
            (after_seq,),
        )
        for seq, _id, ts, sid, typ, payload in cur.fetchall():
            yield (
                int(seq),
                Event(
                    id=str(_id),
                    ts=float(ts),
                    session_id=str(sid),
                    type=str(typ),
                    payload=loads(payload),
                ),
            )

    def close(self) -> None:
        """显式关闭数据库连接。"""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "SQLiteEventLog":
        """上下文管理器入口。"""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """上下文管理器退出 - 确保连接被关闭。"""
        self.close()
