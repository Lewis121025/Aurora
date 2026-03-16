"""身份解析模块。

将会话（session）与长期关系（relation）解耦。
session 是接入通道，relation 是长期对象。多个 session 可绑定同一 relation。

绑定关系持久化在 SQLite identity_bindings 表中。
"""
from __future__ import annotations

import sqlite3
from uuid import uuid4


class IdentityResolver:
    """身份解析器。

    管理 session_id → relation_id 的绑定，支持：
    - 首次 session 自动创建 relation 并绑定
    - 显式绑定 session 到已有 relation
    - 从持久化加载已有绑定

    Attributes:
        _bindings: session_id → relation_id 映射。
        _connection: SQLite 连接（与 persistence 共享）。
    """

    __slots__ = ("_bindings", "_connection")

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._bindings: dict[str, str] = {}
        self._ensure_table()
        self._load()

    def _ensure_table(self) -> None:
        with self._connection:
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS identity_bindings("
                "session_id TEXT PRIMARY KEY, "
                "relation_id TEXT NOT NULL, "
                "created_at REAL NOT NULL)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_bindings_relation "
                "ON identity_bindings(relation_id)"
            )

    def _load(self) -> None:
        rows = self._connection.execute(
            "SELECT session_id, relation_id FROM identity_bindings"
        ).fetchall()
        for row in rows:
            self._bindings[str(row[0])] = str(row[1])

    def resolve(self, session_id: str, now_ts: float) -> str:
        """将 session_id 解析为 relation_id。

        若 session_id 已绑定则返回已有 relation_id。
        否则创建新 relation 并绑定。

        Args:
            session_id: 会话 ID。
            now_ts: 当前时间戳。

        Returns:
            relation_id: 关系 ID。
        """
        if session_id in self._bindings:
            return self._bindings[session_id]
        relation_id = f"rel:{uuid4().hex[:12]}"
        self._bind(session_id, relation_id, now_ts)
        return relation_id

    def bind(self, session_id: str, relation_id: str, now_ts: float) -> None:
        """显式绑定 session 到已有 relation。

        用于合并会话或跨设备识别同一用户。

        Args:
            session_id: 会话 ID。
            relation_id: 目标关系 ID。
            now_ts: 当前时间戳。
        """
        self._bind(session_id, relation_id, now_ts)

    def relation_for(self, session_id: str) -> str | None:
        """查询 session 绑定的 relation_id，不自动创建。"""
        return self._bindings.get(session_id)

    def sessions_for(self, relation_id: str) -> tuple[str, ...]:
        """查询 relation 绑定的所有 session_id。"""
        return tuple(
            sid for sid, rid in self._bindings.items() if rid == relation_id
        )

    def _bind(self, session_id: str, relation_id: str, now_ts: float) -> None:
        self._bindings[session_id] = relation_id
        with self._connection:
            self._connection.execute(
                "INSERT INTO identity_bindings(session_id, relation_id, created_at) "
                "VALUES(?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET relation_id = excluded.relation_id",
                (session_id, relation_id, now_ts),
            )
