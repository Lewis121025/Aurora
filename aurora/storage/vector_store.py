"""
Vector Store Abstraction Layer
==============================

Provides abstract interface for vector storage:
- InMemoryVectorStore: Development/testing
- PgvectorStore: Production PostgreSQL + pgvector with HNSW indexing
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import asyncio
import logging

import numpy as np

from aurora.utils.math_utils import l2_normalize, cosine_sim

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
    """Abstract vector store interface."""
    
    @abstractmethod
    async def add(
        self,
        id: str,
        vec: np.ndarray,
        kind: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        ...
    
    @abstractmethod
    async def batch_add(self, records: Sequence[VectorRecord]) -> int:
        ...
    
    @abstractmethod
    async def remove(self, id: str) -> bool:
        ...
    
    @abstractmethod
    async def batch_remove(self, ids: Sequence[str]) -> int:
        ...
    
    @abstractmethod
    async def get(self, id: str) -> Optional[VectorRecord]:
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
        ...
    
    @abstractmethod
    async def count(
        self,
        kind: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> int:
        ...
    
    async def exists(self, id: str) -> bool:
        return await self.get(id) is not None
    
    async def close(self) -> None:
        pass


class InMemoryVectorStore(VectorStore):
    """In-memory vector store for development and testing."""
    
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
                id=id, vector=vec, kind=kind, user_id=user_id, metadata=metadata or {}
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
                    id=rec.id, vector=vec, kind=rec.kind, user_id=rec.user_id,
                    metadata=rec.metadata, created_at=rec.created_at
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
                if kind is not None and rec.kind != kind:
                    continue
                if user_id is not None and rec.user_id != user_id:
                    continue
                if filters:
                    if any(rec.metadata.get(k) != v for k, v in filters.items()):
                        continue
                hits.append((rec.id, cosine_sim(query, rec.vector)))
            
            hits.sort(key=lambda x: x[1], reverse=True)
            return hits[:k]
    
    async def count(
        self,
        kind: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> int:
        async with self._lock:
            return sum(
                1 for rec in self._records.values()
                if (kind is None or rec.kind == kind)
                and (user_id is None or rec.user_id == user_id)
            )


class PgvectorStore(VectorStore):
    """PostgreSQL + pgvector production vector store with HNSW indexing."""
    
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
        if self._pool is None:
            try:
                import asyncpg
                self._pool = await asyncpg.create_pool(
                    self.dsn, min_size=2, max_size=self.pool_size
                )
            except ImportError:
                raise ImportError("asyncpg required. Install: pip install asyncpg")
    
    async def _init_schema(self) -> None:
        await self._ensure_pool()
        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
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
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_hnsw 
                ON {self.table_name} USING hnsw (embedding vector_cosine_ops) 
                WITH (m = 16, ef_construction = 64)
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
        async with self._pool.acquire() as conn:
            await conn.execute(f"""
                INSERT INTO {self.table_name} (id, user_id, kind, embedding, metadata)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (id) DO UPDATE SET
                    user_id = EXCLUDED.user_id, kind = EXCLUDED.kind,
                    embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata
            """, id, user_id, kind, vec.astype(np.float32).tolist(), metadata or {})
    
    async def batch_add(self, records: Sequence[VectorRecord]) -> int:
        if not records:
            return 0
        await self._ensure_pool()
        async with self._pool.acquire() as conn:
            data = [(r.id, r.user_id, r.kind, r.vector.astype(np.float32).tolist(), r.metadata or {})
                    for r in records]
            await conn.executemany(f"""
                INSERT INTO {self.table_name} (id, user_id, kind, embedding, metadata)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (id) DO UPDATE SET
                    user_id = EXCLUDED.user_id, kind = EXCLUDED.kind,
                    embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata
            """, data)
            return len(records)
    
    async def remove(self, id: str) -> bool:
        await self._ensure_pool()
        async with self._pool.acquire() as conn:
            result = await conn.execute(f"DELETE FROM {self.table_name} WHERE id = $1", id)
            return result == "DELETE 1"
    
    async def batch_remove(self, ids: Sequence[str]) -> int:
        if not ids:
            return 0
        await self._ensure_pool()
        async with self._pool.acquire() as conn:
            result = await conn.execute(f"DELETE FROM {self.table_name} WHERE id = ANY($1)", list(ids))
            try:
                return int(result.split()[-1])
            except (IndexError, ValueError):
                return 0
    
    async def get(self, id: str) -> Optional[VectorRecord]:
        await self._ensure_pool()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(f"""
                SELECT id, user_id, kind, embedding, metadata, created_at
                FROM {self.table_name} WHERE id = $1
            """, id)
            if row is None:
                return None
            return VectorRecord(
                id=row["id"], user_id=row["user_id"], kind=row["kind"],
                vector=np.array(row["embedding"], dtype=np.float32),
                metadata=row["metadata"] or {},
                created_at=row["created_at"].timestamp() if row["created_at"] else None
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
        
        conditions, params = [], [query.astype(np.float32).tolist(), k]
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
        
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT id, 1 - (embedding <=> $1) as similarity
                FROM {self.table_name} {where}
                ORDER BY embedding <=> $1 LIMIT $2
            """, *params)
            return [(row["id"], float(row["similarity"])) for row in rows]
    
    async def count(
        self,
        kind: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> int:
        await self._ensure_pool()
        
        conditions, params = [], []
        param_idx = 1
        if user_id is not None:
            conditions.append(f"user_id = ${param_idx}")
            params.append(user_id)
            param_idx += 1
        if kind is not None:
            conditions.append(f"kind = ${param_idx}")
            params.append(kind)
        
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        
        async with self._pool.acquire() as conn:
            return int(await conn.fetchval(f"SELECT COUNT(*) FROM {self.table_name} {where}", *params))
    
    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None


def create_vector_store(backend: str = "memory", dim: int = 1024, **kwargs) -> VectorStore:
    """Factory to create vector store."""
    if backend == "memory":
        return InMemoryVectorStore(dim=dim)
    elif backend == "pgvector":
        if not kwargs.get("dsn"):
            raise ValueError("dsn required for pgvector")
        return PgvectorStore(
            dsn=kwargs["dsn"], dim=dim,
            table_name=kwargs.get("table_name", "aurora_vectors"),
            pool_size=kwargs.get("pool_size", 10)
        )
    raise ValueError(f"Unknown backend: {backend}")
