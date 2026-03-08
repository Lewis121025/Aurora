from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

from aurora.utils.jsonx import dumps, loads


@dataclass
class Document:
    id: str
    kind: str  # plot/story/theme/self/claim
    ts: float
    body: Dict[str, Any]


class SQLiteDocStore:
    """用于派生工件的简单文档存储。

    表：
      docs(id TEXT PRIMARY KEY, kind TEXT, ts REAL, body TEXT)
    """

    def __init__(self, path: str):
        self.path = path
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS docs ("
            "id TEXT PRIMARY KEY,"
            "kind TEXT,"
            "ts REAL,"
            "body TEXT"
            ")"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_kind_ts ON docs(kind, ts)")
        self._conn.commit()

    def upsert(self, doc: Document) -> None:
        self._conn.execute(
            "INSERT INTO docs(id, kind, ts, body) VALUES(?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET kind=excluded.kind, ts=excluded.ts, body=excluded.body",
            (doc.id, doc.kind, doc.ts, dumps(doc.body)),
        )
        self._conn.commit()

    def get(self, doc_id: str) -> Optional[Document]:
        cur = self._conn.cursor()
        cur.execute("SELECT id, kind, ts, body FROM docs WHERE id = ?", (doc_id,))
        row = cur.fetchone()
        if not row:
            return None
        _id, kind, ts, body = row
        return Document(id=str(_id), kind=str(kind), ts=float(ts), body=loads(body))

    def iter_kind(self, *, kind: str, limit: int = 200) -> Iterable[Document]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT id, kind, ts, body FROM docs WHERE kind = ? ORDER BY ts DESC LIMIT ?",
            (kind, limit),
        )
        for _id, k, ts, body in cur.fetchall():
            yield Document(id=str(_id), kind=str(k), ts=float(ts), body=loads(body))

    def close(self) -> None:
        """显式关闭数据库连接。"""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "SQLiteDocStore":
        """上下文管理器入口。"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器退出 - 确保连接被关闭。"""
        self.close()
