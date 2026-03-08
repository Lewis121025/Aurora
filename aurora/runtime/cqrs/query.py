"""
查询服务 (读取路径)
=========================

处理 CQRS 的读取端:
- 直接查询读优化存储 (向量数据库、缓存)
- 绕过写入路径以实现低延迟
- 支持各种查询类型 (语义、时间、因果)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from aurora.core.models.trace import QueryHit

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    """查询记忆的请求。"""
    user_id: str
    text: str
    k: int = Field(default=8, ge=1, le=50)
    kinds: Optional[List[str]] = None  # ["plot", "story", "theme"]
    include_metadata: bool = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_456",
                "text": "How do I avoid hard-coded thresholds?",
                "k": 8,
                "kinds": ["plot", "story"],
            }
        }


@dataclass
class QueryResponse:
    """查询的响应。"""
    query: str
    hits: List[QueryHit]
    attractor_path_len: int = 0
    latency_ms: float = 0.0
    from_cache: bool = False


class QueryCache:
    """带 TTL 的简单内存查询缓存。"""
    
    def __init__(self, ttl_seconds: float = 60.0, max_size: int = 1000):
        self.ttl = ttl_seconds
        self.max_size = max_size
        self._cache: Dict[str, Tuple[float, QueryResponse]] = {}
        self._lock = asyncio.Lock()
    
    def _cache_key(self, user_id: str, text: str, k: int) -> str:
        return f"{user_id}:{hash(text)}:{k}"
    
    async def get(self, user_id: str, text: str, k: int) -> Optional[QueryResponse]:
        key = self._cache_key(user_id, text, k)
        async with self._lock:
            if key in self._cache:
                ts, response = self._cache[key]
                if time.time() - ts < self.ttl:
                    return response
                else:
                    del self._cache[key]
            return None
    
    async def set(self, user_id: str, text: str, k: int, response: QueryResponse) -> None:
        key = self._cache_key(user_id, text, k)
        async with self._lock:
            # 如果达到容量，驱逐最旧的
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
                del self._cache[oldest_key]
            self._cache[key] = (time.time(), response)
    
    async def invalidate(self, user_id: str) -> int:
        """使用户的所有缓存条目失效。"""
        prefix = f"{user_id}:"
        async with self._lock:
            keys_to_remove = [k for k in self._cache if k.startswith(prefix)]
            for key in keys_to_remove:
                del self._cache[key]
            return len(keys_to_remove)


class QueryService:
    """CQRS 架构的读取服务。

    直接查询读优化存储:
    - 用于语义搜索的向量存储
    - 用于重复查询的缓存
    - 用于元数据的内存状态

    不经过写入路径。
    """
    
    def __init__(
        self,
        vector_store=None,
        state_store=None,
        memory=None,  # AuroraMemory for fallback/sync mode
        cache: Optional[QueryCache] = None,
        cache_enabled: bool = True,
    ):
        """初始化查询服务。

        参数:
            vector_store: 用于语义搜索的 VectorStore 实例
            state_store: 用于元数据的 StateStore 实例
            memory: 用于同步/回退模式的 AuroraMemory 实例
            cache: 查询缓存 (默认: 创建新缓存)
            cache_enabled: 是否使用缓存
        """
        self.vector_store = vector_store
        self.state_store = state_store
        self.memory = memory
        self.cache = cache or QueryCache() if cache_enabled else None
        self.cache_enabled = cache_enabled

        # 统计信息
        self._total_queries = 0
        self._cache_hits = 0
        self._cache_misses = 0
    
    async def query(self, request: QueryRequest) -> QueryResponse:
        """执行语义查询。

        按顺序尝试:
        1. 缓存查找
        2. 向量存储搜索
        3. 回退到内存搜索
        """
        start_time = time.time()
        self._total_queries += 1

        # 先尝试缓存
        if self.cache is not None:
            cached = await self.cache.get(request.user_id, request.text, request.k)
            if cached is not None:
                self._cache_hits += 1
                cached.from_cache = True
                return cached

        self._cache_misses += 1

        # 执行查询
        if self.vector_store is not None:
            response = await self._query_vector_store(request)
        elif self.memory is not None:
            response = await self._query_memory_sync(request)
        else:
            response = QueryResponse(
                query=request.text,
                hits=[],
                attractor_path_len=0,
            )

        # 计算延迟
        response.latency_ms = (time.time() - start_time) * 1000

        # 缓存结果
        if self.cache is not None:
            await self.cache.set(request.user_id, request.text, request.k, response)

        return response
    
    async def _query_vector_store(self, request: QueryRequest) -> QueryResponse:
        """使用向量存储进行查询。"""
        # 这需要嵌入提供者来嵌入查询
        # 目前，我们将使用简化的方法

        # TODO: 将嵌入提供者添加到 QueryService
        # query_embedding = await self.embedding_provider.embed(request.text)

        # 目前回退到基于内存的查询
        if self.memory is not None:
            return await self._query_memory_sync(request)
        
        return QueryResponse(
            query=request.text,
            hits=[],
            attractor_path_len=0,
        )
    
    async def _query_memory_sync(self, request: QueryRequest) -> QueryResponse:
        """使用内存中的 AuroraMemory 进行查询 (同步模式)。"""
        if self.memory is None:
            return QueryResponse(query=request.text, hits=[], attractor_path_len=0)

        # 在线程池中运行同步查询
        def _sync_query():
            return self.memory.query(request.text, k=request.k)

        loop = asyncio.get_event_loop()
        trace = await loop.run_in_executor(None, _sync_query)

        hits = []
        kinds_filter = set(request.kinds) if request.kinds else None

        for nid, score, kind in trace.ranked:
            # 应用类型过滤
            if kinds_filter and kind not in kinds_filter:
                continue

            # 获取片段
            snippet = ""
            if kind == "plot":
                plot = self.memory.plots.get(nid)
                if plot:
                    snippet = plot.text[:240] + "..." if len(plot.text) > 240 else plot.text
            elif kind == "story":
                story = self.memory.stories.get(nid)
                if story:
                    snippet = f"Story with {len(story.plot_ids)} plots"
            elif kind == "theme":
                theme = self.memory.themes.get(nid)
                if theme:
                    snippet = theme.description or theme.name or f"Theme with {len(theme.story_ids)} stories"

            metadata = None
            if request.include_metadata:
                metadata = await self._get_metadata(nid, kind)

            hits.append(QueryHit(
                id=nid,
                kind=kind,
                score=float(score),
                snippet=snippet,
                metadata=metadata,
            ))

        return QueryResponse(
            query=request.text,
            hits=hits,
            attractor_path_len=len(trace.attractor_path),
        )
    
    async def _get_metadata(self, node_id: str, kind: str) -> Dict[str, Any]:
        """获取节点的元数据。"""
        if self.memory is None:
            return {}
        
        if kind == "plot":
            plot = self.memory.plots.get(node_id)
            if plot:
                return {
                    "ts": plot.ts,
                    "actors": list(plot.actors),
                    "tension": plot.tension,
                    "surprise": plot.surprise,
                    "story_id": plot.story_id,
                    "status": plot.status,
                }
        elif kind == "story":
            story = self.memory.stories.get(node_id)
            if story:
                return {
                    "created_ts": story.created_ts,
                    "updated_ts": story.updated_ts,
                    "plot_count": len(story.plot_ids),
                    "status": story.status,
                    "actor_counts": story.actor_counts,
                }
        elif kind == "theme":
            theme = self.memory.themes.get(node_id)
            if theme:
                return {
                    "created_ts": theme.created_ts,
                    "updated_ts": theme.updated_ts,
                    "story_count": len(theme.story_ids),
                    "confidence": theme.confidence(),
                    "type": theme.theme_type,
                }
        
        return {}
    
    async def invalidate_cache(self, user_id: str) -> int:
        """使用户的缓存条目失效。"""
        if self.cache is not None:
            return await self.cache.invalidate(user_id)
        return 0
    
    def stats(self) -> Dict[str, Any]:
        """返回服务统计信息。"""
        cache_rate = self._cache_hits / max(1, self._total_queries)
        return {
            "total_queries": self._total_queries,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": cache_rate,
            "cache_enabled": self.cache_enabled,
        }
