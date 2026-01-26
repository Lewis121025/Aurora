"""
Vector Store Abstraction Layer
==============================

Provides abstract interface for vector storage with multiple implementations:
- InMemoryVectorStore: Development/testing (replaces VectorIndex from aurora_core)
- PgvectorStore: Production PostgreSQL + pgvector with HNSW indexing
- MilvusStore: Alternative production option (placeholder)

Design goals:
- Sub-millisecond retrieval with HNSW indexes
- Metadata filtering (user_id sharding)
- Async-first API for high concurrency
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import asyncio
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VectorRecord:
    """A vector record with metadata."""
    id: str
    vector: np.ndarray
    kind: str
    user_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[float] = None


class VectorStore(ABC):
    """Abstract vector store interface.
    
    All implementations should support:
    - CRUD operations on vectors
    - Similarity search with optional filtering
    - Batch operations for efficiency
    """
    
    @abstractmethod
    async def add(
        self,
        id: str,
        vec: np.ndarray,
        kind: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a single vector to the store."""
        ...
    
    @abstractmethod
    async def batch_add(
        self,
        records: Sequence[VectorRecord],
    ) -> int:
        """Add multiple vectors in batch. Returns count of added records."""
        ...
    
    @abstractmethod
    async def remove(self, id: str) -> bool:
        """Remove a vector by ID. Returns True if found and removed."""
        ...
    
    @abstractmethod
    async def batch_remove(self, ids: Sequence[str]) -> int:
        """Remove multiple vectors. Returns count of removed records."""
        ...
    
    @abstractmethod
    async def get(self, id: str) -> Optional[VectorRecord]:
        """Get a vector record by ID."""
        ...
    
    @abstractmethod
    async def search(
        self,
        query: np.ndarray,
        k: int = 10,
        kind: Optional[str] = None,
        user_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        """Search for similar vectors.
        
        Args:
            query: Query vector
            k: Number of results to return
            kind: Filter by kind (plot, story, theme)
            user_id: Filter by user_id (for multi-tenant sharding)
            filters: Additional metadata filters
            
        Returns:
            List of (id, similarity_score) tuples, sorted by score descending
        """
        ...
    
    @abstractmethod
    async def count(
        self,
        kind: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> int:
        """Count vectors matching the filters."""
        ...
    
    async def exists(self, id: str) -> bool:
        """Check if a vector exists."""
        return await self.get(id) is not None
    
    async def close(self) -> None:
        """Close connections and cleanup resources."""
        pass


def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize vector to unit length."""
    n = float(np.linalg.norm(v))
    if n < eps:
        return v.copy()
    return (v / n).astype(np.float32)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(_l2_normalize(a), _l2_normalize(b)))


class InMemoryVectorStore(VectorStore):
    """In-memory vector store for development and testing.
    
    This is a direct replacement for VectorIndex from aurora_core.py,
    but with async interface and metadata support.
    """
    
    def __init__(self, dim: int):
        self.dim = dim
        self._records: Dict[str, VectorRecord] = {}
        self._lock = asyncio.Lock()
    
    async def add(
        self,
        id: str,
        vec: np.ndarray,
        kind: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        vec = vec.astype(np.float32)
        if vec.shape != (self.dim,):
            raise ValueError(f"Vector dimension mismatch: {vec.shape} vs ({self.dim},)")
        
        async with self._lock:
            self._records[id] = VectorRecord(
                id=id,
                vector=vec,
                kind=kind,
                user_id=user_id,
                metadata=metadata or {},
            )
    
    async def batch_add(self, records: Sequence[VectorRecord]) -> int:
        async with self._lock:
            count = 0
            for rec in records:
                vec = rec.vector.astype(np.float32)
                if vec.shape != (self.dim,):
                    logger.warning(f"Skipping record {rec.id}: dimension mismatch")
                    continue
                self._records[rec.id] = VectorRecord(
                    id=rec.id,
                    vector=vec,
                    kind=rec.kind,
                    user_id=rec.user_id,
                    metadata=rec.metadata,
                    created_at=rec.created_at,
                )
                count += 1
            return count
    
    async def remove(self, id: str) -> bool:
        async with self._lock:
            if id in self._records:
                del self._records[id]
                return True
            return False
    
    async def batch_remove(self, ids: Sequence[str]) -> int:
        async with self._lock:
            count = 0
            for id in ids:
                if id in self._records:
                    del self._records[id]
                    count += 1
            return count
    
    async def get(self, id: str) -> Optional[VectorRecord]:
        async with self._lock:
            return self._records.get(id)
    
    async def search(
        self,
        query: np.ndarray,
        k: int = 10,
        kind: Optional[str] = None,
        user_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        query = query.astype(np.float32)
        
        async with self._lock:
            hits: List[Tuple[str, float]] = []
            
            for rec in self._records.values():
                # Apply filters
                if kind is not None and rec.kind != kind:
                    continue
                if user_id is not None and rec.user_id != user_id:
                    continue
                if filters:
                    skip = False
                    for key, value in filters.items():
                        if rec.metadata.get(key) != value:
                            skip = True
                            break
                    if skip:
                        continue
                
                # Compute similarity
                sim = _cosine_sim(query, rec.vector)
                hits.append((rec.id, sim))
            
            # Sort by similarity descending
            hits.sort(key=lambda x: x[1], reverse=True)
            return hits[:k]
    
    async def count(
        self,
        kind: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> int:
        async with self._lock:
            count = 0
            for rec in self._records.values():
                if kind is not None and rec.kind != kind:
                    continue
                if user_id is not None and rec.user_id != user_id:
                    continue
                count += 1
            return count
    
    def sync_add(self, id: str, vec: np.ndarray, kind: str, user_id: str = "") -> None:
        """Synchronous add for backward compatibility with VectorIndex."""
        vec = vec.astype(np.float32)
        if vec.shape != (self.dim,):
            raise ValueError(f"Vector dimension mismatch: {vec.shape} vs ({self.dim},)")
        self._records[id] = VectorRecord(id=id, vector=vec, kind=kind, user_id=user_id)
    
    def sync_remove(self, id: str) -> None:
        """Synchronous remove for backward compatibility."""
        self._records.pop(id, None)
    
    def sync_search(
        self,
        query: np.ndarray,
        k: int = 10,
        kind: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Synchronous search for backward compatibility."""
        query = query.astype(np.float32)
        hits: List[Tuple[str, float]] = []
        
        for rec in self._records.values():
            if kind is not None and rec.kind != kind:
                continue
            sim = _cosine_sim(query, rec.vector)
            hits.append((rec.id, sim))
        
        hits.sort(key=lambda x: x[1], reverse=True)
        return hits[:k]


class PgvectorStore(VectorStore):
    """PostgreSQL + pgvector production vector store.
    
    Features:
    - HNSW indexing for sub-millisecond retrieval
    - Metadata filtering via JSONB
    - User sharding via user_id column
    
    Requires:
    - PostgreSQL 15+ with pgvector extension
    - asyncpg for async connections
    
    Schema:
    ```sql
    CREATE EXTENSION IF NOT EXISTS vector;
    
    CREATE TABLE aurora_vectors (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        embedding vector(1024),
        metadata JSONB DEFAULT '{}',
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    
    CREATE INDEX idx_aurora_vectors_hnsw ON aurora_vectors 
        USING hnsw (embedding vector_cosine_ops) 
        WITH (m = 16, ef_construction = 64);
    
    CREATE INDEX idx_aurora_vectors_user ON aurora_vectors (user_id);
    CREATE INDEX idx_aurora_vectors_kind ON aurora_vectors (kind);
    CREATE INDEX idx_aurora_vectors_user_kind ON aurora_vectors (user_id, kind);
    ```
    """
    
    def __init__(
        self,
        dsn: str,
        dim: int = 1024,
        table_name: str = "aurora_vectors",
        pool_size: int = 10,
    ):
        self.dsn = dsn
        self.dim = dim
        self.table_name = table_name
        self.pool_size = pool_size
        self._pool = None
    
    async def _ensure_pool(self):
        """Lazily create connection pool."""
        if self._pool is None:
            try:
                import asyncpg
                self._pool = await asyncpg.create_pool(
                    self.dsn,
                    min_size=2,
                    max_size=self.pool_size,
                )
            except ImportError:
                raise ImportError("asyncpg is required for PgvectorStore. Install with: pip install asyncpg")
    
    async def _init_schema(self) -> None:
        """Initialize database schema."""
        await self._ensure_pool()
        
        async with self._pool.acquire() as conn:
            # Create extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    embedding vector({self.dim}),
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # Create indexes (idempotent)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_hnsw 
                ON {self.table_name} 
                USING hnsw (embedding vector_cosine_ops) 
                WITH (m = 16, ef_construction = 64)
            """)
            
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_user 
                ON {self.table_name} (user_id)
            """)
            
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_kind 
                ON {self.table_name} (kind)
            """)
            
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_user_kind 
                ON {self.table_name} (user_id, kind)
            """)
    
    async def add(
        self,
        id: str,
        vec: np.ndarray,
        kind: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        await self._ensure_pool()
        
        vec = vec.astype(np.float32)
        vec_list = vec.tolist()
        meta_json = metadata or {}
        
        async with self._pool.acquire() as conn:
            await conn.execute(f"""
                INSERT INTO {self.table_name} (id, user_id, kind, embedding, metadata)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (id) DO UPDATE SET
                    user_id = EXCLUDED.user_id,
                    kind = EXCLUDED.kind,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
            """, id, user_id, kind, vec_list, meta_json)
    
    async def batch_add(self, records: Sequence[VectorRecord]) -> int:
        await self._ensure_pool()
        
        if not records:
            return 0
        
        async with self._pool.acquire() as conn:
            # Use COPY for efficiency
            data = [
                (rec.id, rec.user_id, rec.kind, rec.vector.astype(np.float32).tolist(), rec.metadata or {})
                for rec in records
            ]
            
            # Use executemany for batch insert
            await conn.executemany(f"""
                INSERT INTO {self.table_name} (id, user_id, kind, embedding, metadata)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (id) DO UPDATE SET
                    user_id = EXCLUDED.user_id,
                    kind = EXCLUDED.kind,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
            """, data)
            
            return len(records)
    
    async def remove(self, id: str) -> bool:
        await self._ensure_pool()
        
        async with self._pool.acquire() as conn:
            result = await conn.execute(f"""
                DELETE FROM {self.table_name} WHERE id = $1
            """, id)
            return result == "DELETE 1"
    
    async def batch_remove(self, ids: Sequence[str]) -> int:
        await self._ensure_pool()
        
        if not ids:
            return 0
        
        async with self._pool.acquire() as conn:
            result = await conn.execute(f"""
                DELETE FROM {self.table_name} WHERE id = ANY($1)
            """, list(ids))
            # Parse "DELETE N" result
            try:
                return int(result.split()[-1])
            except (IndexError, ValueError):
                return 0
    
    async def get(self, id: str) -> Optional[VectorRecord]:
        await self._ensure_pool()
        
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(f"""
                SELECT id, user_id, kind, embedding, metadata, created_at
                FROM {self.table_name}
                WHERE id = $1
            """, id)
            
            if row is None:
                return None
            
            return VectorRecord(
                id=row["id"],
                user_id=row["user_id"],
                kind=row["kind"],
                vector=np.array(row["embedding"], dtype=np.float32),
                metadata=row["metadata"] or {},
                created_at=row["created_at"].timestamp() if row["created_at"] else None,
            )
    
    async def search(
        self,
        query: np.ndarray,
        k: int = 10,
        kind: Optional[str] = None,
        user_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        await self._ensure_pool()
        
        query = query.astype(np.float32)
        query_list = query.tolist()
        
        # Build query with filters
        conditions = []
        params = [query_list, k]
        param_idx = 3
        
        if user_id is not None:
            conditions.append(f"user_id = ${param_idx}")
            params.append(user_id)
            param_idx += 1
        
        if kind is not None:
            conditions.append(f"kind = ${param_idx}")
            params.append(kind)
            param_idx += 1
        
        if filters:
            for key, value in filters.items():
                conditions.append(f"metadata->>'{key}' = ${param_idx}")
                params.append(str(value))
                param_idx += 1
        
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        
        async with self._pool.acquire() as conn:
            # Use cosine distance operator <=> for similarity
            # Note: pgvector returns distance, we convert to similarity
            rows = await conn.fetch(f"""
                SELECT id, 1 - (embedding <=> $1) as similarity
                FROM {self.table_name}
                {where_clause}
                ORDER BY embedding <=> $1
                LIMIT $2
            """, *params)
            
            return [(row["id"], float(row["similarity"])) for row in rows]
    
    async def count(
        self,
        kind: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> int:
        await self._ensure_pool()
        
        conditions = []
        params = []
        param_idx = 1
        
        if user_id is not None:
            conditions.append(f"user_id = ${param_idx}")
            params.append(user_id)
            param_idx += 1
        
        if kind is not None:
            conditions.append(f"kind = ${param_idx}")
            params.append(kind)
            param_idx += 1
        
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        
        async with self._pool.acquire() as conn:
            result = await conn.fetchval(f"""
                SELECT COUNT(*) FROM {self.table_name} {where_clause}
            """, *params)
            return int(result)
    
    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


class MilvusStore(VectorStore):
    """Milvus vector store implementation (placeholder).
    
    For future implementation when Milvus is needed.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "aurora_vectors",
        dim: int = 1024,
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dim = dim
        raise NotImplementedError("MilvusStore is not yet implemented. Use PgvectorStore instead.")
    
    async def add(self, id: str, vec: np.ndarray, kind: str, user_id: str, metadata: Optional[Dict] = None) -> None:
        raise NotImplementedError()
    
    async def batch_add(self, records: Sequence[VectorRecord]) -> int:
        raise NotImplementedError()
    
    async def remove(self, id: str) -> bool:
        raise NotImplementedError()
    
    async def batch_remove(self, ids: Sequence[str]) -> int:
        raise NotImplementedError()
    
    async def get(self, id: str) -> Optional[VectorRecord]:
        raise NotImplementedError()
    
    async def search(
        self,
        query: np.ndarray,
        k: int = 10,
        kind: Optional[str] = None,
        user_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        raise NotImplementedError()
    
    async def count(self, kind: Optional[str] = None, user_id: Optional[str] = None) -> int:
        raise NotImplementedError()


# Factory function
def create_vector_store(
    backend: str = "memory",
    dim: int = 1024,
    **kwargs,
) -> VectorStore:
    """Factory function to create vector store.
    
    Args:
        backend: "memory", "pgvector", or "milvus"
        dim: Vector dimension
        **kwargs: Backend-specific arguments
        
    Returns:
        VectorStore instance
    """
    if backend == "memory":
        return InMemoryVectorStore(dim=dim)
    elif backend == "pgvector":
        dsn = kwargs.get("dsn")
        if not dsn:
            raise ValueError("dsn is required for pgvector backend")
        return PgvectorStore(
            dsn=dsn,
            dim=dim,
            table_name=kwargs.get("table_name", "aurora_vectors"),
            pool_size=kwargs.get("pool_size", 10),
        )
    elif backend == "milvus":
        return MilvusStore(
            host=kwargs.get("host", "localhost"),
            port=kwargs.get("port", 19530),
            collection_name=kwargs.get("collection_name", "aurora_vectors"),
            dim=dim,
        )
    else:
        raise ValueError(f"Unknown vector store backend: {backend}")
