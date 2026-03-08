"""
Aurora 存储层
====================

提供当前单用户运行时实际使用的本地持久化组件：
- 事件日志（SQLiteEventLog）：仅追加事件溯源
- 文档存储（SQLiteDocStore）：结构化派生文档
- 快照存储（SnapshotStore）：本地 pickle 快照
"""

from aurora.integrations.storage.event_log import Event, SQLiteEventLog
from aurora.integrations.storage.doc_store import Document, SQLiteDocStore
from aurora.integrations.storage.snapshot import Snapshot, SnapshotStore

__all__ = [
    "Event",
    "SQLiteEventLog",
    "Document",
    "SQLiteDocStore",
    "Snapshot",
    "SnapshotStore",
]
