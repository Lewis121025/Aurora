"""
Aurora 存储层
====================

提供以下存储抽象：
- 向量存储（VectorStore）：具有相似性搜索的语义嵌入
- 状态存储（StateStore）：热/冷分层状态管理
- 事件日志（SQLiteEventLog）：仅追加事件溯源
- 文档存储（SQLiteDocStore）：结构化文档存储
"""

from aurora.integrations.storage.event_log import Event, SQLiteEventLog
from aurora.integrations.storage.doc_store import Document, SQLiteDocStore
from aurora.integrations.storage.vector_store import (
    VectorStore,
    VectorRecord,
    InMemoryVectorStore,
    PgvectorStore,
    create_vector_store,
)
from aurora.integrations.storage.state_store import (
    StateStore,
    InMemoryStateStore,
    RedisPostgresStateStore,
    SQLiteStateStore,
    create_state_store,
)

__all__ = [
    "Event",
    "SQLiteEventLog",
    "Document",
    "SQLiteDocStore",
    "VectorStore",
    "VectorRecord",
    "InMemoryVectorStore",
    "PgvectorStore",
    "create_vector_store",
    "StateStore",
    "InMemoryStateStore",
    "RedisPostgresStateStore",
    "SQLiteStateStore",
    "create_state_store",
]
