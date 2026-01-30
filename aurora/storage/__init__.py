"""
Aurora Storage Layer
====================

Provides storage abstractions for:
- Vector storage (VectorStore): Semantic embeddings with similarity search
- State storage (StateStore): Hot/cold tiered state management
- Event logging (SQLiteEventLog): Append-only event sourcing
- Document storage (SQLiteDocStore): Structured document storage
"""

from aurora.storage.event_log import Event, SQLiteEventLog
from aurora.storage.doc_store import Document, SQLiteDocStore
from aurora.storage.vector_store import (
    VectorStore,
    VectorRecord,
    InMemoryVectorStore,
    PgvectorStore,
    create_vector_store,
)
from aurora.storage.state_store import (
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
