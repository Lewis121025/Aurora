"""
状态存储抽象层
=============================

为算法状态提供热/冷数据存储：
- 热数据：活跃故事、最近情节 -> Redis（快速读取）
- 冷数据：归档情节、旧主题 -> PostgreSQL JSONB

设计目标：
- 用结构化 JSON 序列化替换基于 pickle 的快照
- 支持算法类的状态字典序列化
- 启用具有共享状态存储的水平扩展
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import asyncio
import json
import logging
import time

logger = logging.getLogger(__name__)


class StateStore(ABC):
    """抽象状态存储接口。

    提供热/冷层存储：
    - 热：低延迟读取（Redis/内存）
    - 冷：持久化、可查询存储（PostgreSQL/SQLite）
    """

    @abstractmethod
    async def save_hot(self, user_id: str, key: str, value: Dict[str, Any]) -> None:
        """保存状态到热存储（快速，可能是临时的）。"""
        ...

    @abstractmethod
    async def load_hot(self, user_id: str, key: str) -> Optional[Dict[str, Any]]:
        """从热存储加载状态。"""
        ...

    @abstractmethod
    async def delete_hot(self, user_id: str, key: str) -> bool:
        """从热存储删除。如果存在则返回 True。"""
        ...

    @abstractmethod
    async def save_cold(self, user_id: str, key: str, value: Dict[str, Any]) -> None:
        """保存状态到冷存储（持久化）。"""
        ...

    @abstractmethod
    async def load_cold(self, user_id: str, key: str) -> Optional[Dict[str, Any]]:
        """从冷存储加载状态。"""
        ...

    @abstractmethod
    async def delete_cold(self, user_id: str, key: str) -> bool:
        """从冷存储删除。如果存在则返回 True。"""
        ...

    @abstractmethod
    async def list_keys(self, user_id: str, prefix: str = "", tier: str = "hot") -> List[str]:
        """列出指定层中与前缀匹配的键。"""
        ...
    
    async def save_snapshot(self, user_id: str, seq: int, state_dict: Dict[str, Any]) -> None:
        """保存快照（进入冷存储，使用特殊键）。"""
        key = f"snapshot:{seq}"
        await self.save_cold(user_id, key, {
            "seq": seq,
            "state": state_dict,
            "saved_at": time.time(),
        })

    async def load_latest_snapshot(self, user_id: str) -> Optional[Tuple[int, Dict[str, Any]]]:
        """加载最新快照。返回 (seq, state_dict) 或 None。"""
        keys = await self.list_keys(user_id, prefix="snapshot:", tier="cold")
        if not keys:
            return None

        # 查找最高的序列号
        best_seq = -1
        best_key = None
        for key in keys:
            try:
                seq = int(key.split(":")[-1])
                if seq > best_seq:
                    best_seq = seq
                    best_key = key
            except ValueError:
                continue

        if best_key is None:
            return None

        data = await self.load_cold(user_id, best_key)
        if data is None:
            return None

        return data.get("seq", best_seq), data.get("state", {})

    async def close(self) -> None:
        """关闭连接并清理资源。"""
        pass


class InMemoryStateStore(StateStore):
    """用于开发和测试的内存状态存储。"""

    def __init__(self):
        self._hot: Dict[str, Dict[str, Dict[str, Any]]] = {}  # user_id -> key -> value
        self._cold: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._lock = asyncio.Lock()
    
    async def save_hot(self, user_id: str, key: str, value: Dict[str, Any]) -> None:
        async with self._lock:
            if user_id not in self._hot:
                self._hot[user_id] = {}
            self._hot[user_id][key] = value
    
    async def load_hot(self, user_id: str, key: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return self._hot.get(user_id, {}).get(key)
    
    async def delete_hot(self, user_id: str, key: str) -> bool:
        async with self._lock:
            if user_id in self._hot and key in self._hot[user_id]:
                del self._hot[user_id][key]
                return True
            return False
    
    async def save_cold(self, user_id: str, key: str, value: Dict[str, Any]) -> None:
        async with self._lock:
            if user_id not in self._cold:
                self._cold[user_id] = {}
            self._cold[user_id][key] = value
    
    async def load_cold(self, user_id: str, key: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return self._cold.get(user_id, {}).get(key)
    
    async def delete_cold(self, user_id: str, key: str) -> bool:
        async with self._lock:
            if user_id in self._cold and key in self._cold[user_id]:
                del self._cold[user_id][key]
                return True
            return False
    
    async def list_keys(self, user_id: str, prefix: str = "", tier: str = "hot") -> List[str]:
        async with self._lock:
            store = self._hot if tier == "hot" else self._cold
            user_store = store.get(user_id, {})
            return [k for k in user_store.keys() if k.startswith(prefix)]


class RedisPostgresStateStore(StateStore):
    """生产级状态存储，使用 Redis（热）+ PostgreSQL（冷）。

    热层（Redis）：
    - 活跃内存状态
    - 最近查询缓存
    - 会话数据

    冷层（PostgreSQL）：
    - 快照
    - 归档状态
    - 历史数据

    PostgreSQL 的模式：
    ```sql
    CREATE TABLE aurora_states (
        id SERIAL PRIMARY KEY,
        user_id TEXT NOT NULL,
        key TEXT NOT NULL,
        value JSONB NOT NULL,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW(),
        UNIQUE(user_id, key)
    );

    CREATE INDEX idx_aurora_states_user ON aurora_states (user_id);
    CREATE INDEX idx_aurora_states_user_key ON aurora_states (user_id, key);
    ```
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        postgres_dsn: Optional[str] = None,
        redis_prefix: str = "aurora:state:",
        pg_table: str = "aurora_states",
        redis_ttl: int = 3600 * 24,  # 24 hours default TTL
    ):
        self.redis_url = redis_url
        self.postgres_dsn = postgres_dsn
        self.redis_prefix = redis_prefix
        self.pg_table = pg_table
        self.redis_ttl = redis_ttl
        
        self._redis = None
        self._pg_pool = None
    
    async def _ensure_redis(self):
        """延迟创建 Redis 连接。"""
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
            except ImportError:
                raise ImportError("redis is required for RedisPostgresStateStore. Install with: pip install redis")

    async def _ensure_postgres(self):
        """延迟创建 PostgreSQL 连接池。"""
        if self._pg_pool is None and self.postgres_dsn:
            try:
                import asyncpg
                self._pg_pool = await asyncpg.create_pool(
                    self.postgres_dsn,
                    min_size=2,
                    max_size=10,
                )
            except ImportError:
                raise ImportError("asyncpg is required for RedisPostgresStateStore. Install with: pip install asyncpg")

    def _redis_key(self, user_id: str, key: str) -> str:
        """构建 Redis 键。"""
        return f"{self.redis_prefix}{user_id}:{key}"
    
    async def save_hot(self, user_id: str, key: str, value: Dict[str, Any]) -> None:
        await self._ensure_redis()
        redis_key = self._redis_key(user_id, key)
        await self._redis.setex(redis_key, self.redis_ttl, json.dumps(value))
    
    async def load_hot(self, user_id: str, key: str) -> Optional[Dict[str, Any]]:
        await self._ensure_redis()
        redis_key = self._redis_key(user_id, key)
        data = await self._redis.get(redis_key)
        if data is None:
            return None
        return json.loads(data)
    
    async def delete_hot(self, user_id: str, key: str) -> bool:
        await self._ensure_redis()
        redis_key = self._redis_key(user_id, key)
        result = await self._redis.delete(redis_key)
        return result > 0
    
    async def save_cold(self, user_id: str, key: str, value: Dict[str, Any]) -> None:
        await self._ensure_postgres()
        
        if self._pg_pool is None:
            logger.warning("PostgreSQL not configured, cold storage unavailable")
            return
        
        async with self._pg_pool.acquire() as conn:
            await conn.execute(f"""
                INSERT INTO {self.pg_table} (user_id, key, value, updated_at)
                VALUES ($1, $2, $3, NOW())
                ON CONFLICT (user_id, key) DO UPDATE SET
                    value = EXCLUDED.value,
                    updated_at = NOW()
            """, user_id, key, json.dumps(value))
    
    async def load_cold(self, user_id: str, key: str) -> Optional[Dict[str, Any]]:
        await self._ensure_postgres()
        
        if self._pg_pool is None:
            return None
        
        async with self._pg_pool.acquire() as conn:
            row = await conn.fetchrow(f"""
                SELECT value FROM {self.pg_table}
                WHERE user_id = $1 AND key = $2
            """, user_id, key)
            
            if row is None:
                return None
            
            return json.loads(row["value"])
    
    async def delete_cold(self, user_id: str, key: str) -> bool:
        await self._ensure_postgres()
        
        if self._pg_pool is None:
            return False
        
        async with self._pg_pool.acquire() as conn:
            result = await conn.execute(f"""
                DELETE FROM {self.pg_table}
                WHERE user_id = $1 AND key = $2
            """, user_id, key)
            return result == "DELETE 1"
    
    async def list_keys(self, user_id: str, prefix: str = "", tier: str = "hot") -> List[str]:
        if tier == "hot":
            await self._ensure_redis()
            pattern = self._redis_key(user_id, f"{prefix}*")
            keys = await self._redis.keys(pattern)
            # 去掉前缀以返回仅键部分
            base_prefix = self._redis_key(user_id, "")
            return [k[len(base_prefix):] for k in keys]
        else:
            await self._ensure_postgres()
            if self._pg_pool is None:
                return []

            async with self._pg_pool.acquire() as conn:
                rows = await conn.fetch(f"""
                    SELECT key FROM {self.pg_table}
                    WHERE user_id = $1 AND key LIKE $2
                """, user_id, f"{prefix}%")
                return [row["key"] for row in rows]

    async def _init_schema(self) -> None:
        """初始化 PostgreSQL 模式。"""
        await self._ensure_postgres()

        if self._pg_pool is None:
            return

        async with self._pg_pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.pg_table} (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value JSONB NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(user_id, key)
                )
            """)

            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.pg_table}_user
                ON {self.pg_table} (user_id)
            """)

            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.pg_table}_user_key
                ON {self.pg_table} (user_id, key)
            """)
    
    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.close()
            self._redis = None
        
        if self._pg_pool is not None:
            await self._pg_pool.close()
            self._pg_pool = None


class SQLiteStateStore(StateStore):
    """用于单节点部署的基于 SQLite 的状态存储。

    使用单个 SQLite 数据库，为热/冷层分别使用表。
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = None
        self._lock = asyncio.Lock()

    async def _ensure_conn(self):
        """延迟创建 SQLite 连接。"""
        if self._conn is None:
            import aiosqlite
            self._conn = await aiosqlite.connect(self.db_path)
            await self._init_schema()

    async def _init_schema(self) -> None:
        """初始化 SQLite 模式。"""
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS hot_states (
                user_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at REAL DEFAULT (unixepoch()),
                PRIMARY KEY (user_id, key)
            )
        """)
        
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cold_states (
                user_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at REAL DEFAULT (unixepoch()),
                updated_at REAL DEFAULT (unixepoch()),
                PRIMARY KEY (user_id, key)
            )
        """)
        
        await self._conn.commit()
    
    async def save_hot(self, user_id: str, key: str, value: Dict[str, Any]) -> None:
        async with self._lock:
            await self._ensure_conn()
            await self._conn.execute("""
                INSERT OR REPLACE INTO hot_states (user_id, key, value, updated_at)
                VALUES (?, ?, ?, unixepoch())
            """, (user_id, key, json.dumps(value)))
            await self._conn.commit()
    
    async def load_hot(self, user_id: str, key: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            await self._ensure_conn()
            cursor = await self._conn.execute("""
                SELECT value FROM hot_states WHERE user_id = ? AND key = ?
            """, (user_id, key))
            row = await cursor.fetchone()
            if row is None:
                return None
            return json.loads(row[0])
    
    async def delete_hot(self, user_id: str, key: str) -> bool:
        async with self._lock:
            await self._ensure_conn()
            cursor = await self._conn.execute("""
                DELETE FROM hot_states WHERE user_id = ? AND key = ?
            """, (user_id, key))
            await self._conn.commit()
            return cursor.rowcount > 0
    
    async def save_cold(self, user_id: str, key: str, value: Dict[str, Any]) -> None:
        async with self._lock:
            await self._ensure_conn()
            await self._conn.execute("""
                INSERT OR REPLACE INTO cold_states (user_id, key, value, updated_at)
                VALUES (?, ?, ?, unixepoch())
            """, (user_id, key, json.dumps(value)))
            await self._conn.commit()
    
    async def load_cold(self, user_id: str, key: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            await self._ensure_conn()
            cursor = await self._conn.execute("""
                SELECT value FROM cold_states WHERE user_id = ? AND key = ?
            """, (user_id, key))
            row = await cursor.fetchone()
            if row is None:
                return None
            return json.loads(row[0])
    
    async def delete_cold(self, user_id: str, key: str) -> bool:
        async with self._lock:
            await self._ensure_conn()
            cursor = await self._conn.execute("""
                DELETE FROM cold_states WHERE user_id = ? AND key = ?
            """, (user_id, key))
            await self._conn.commit()
            return cursor.rowcount > 0
    
    async def list_keys(self, user_id: str, prefix: str = "", tier: str = "hot") -> List[str]:
        async with self._lock:
            await self._ensure_conn()
            table = "hot_states" if tier == "hot" else "cold_states"
            cursor = await self._conn.execute(f"""
                SELECT key FROM {table}
                WHERE user_id = ? AND key LIKE ?
            """, (user_id, f"{prefix}%"))
            rows = await cursor.fetchall()
            return [row[0] for row in rows]
    
    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None:
                await self._conn.close()
                self._conn = None


# 工厂函数
def create_state_store(
    backend: str = "memory",
    **kwargs,
) -> StateStore:
    """工厂函数用于创建状态存储。

    参数：
        backend: "memory"、"sqlite" 或 "redis_postgres"
        **kwargs: 后端特定的参数

    返回：
        StateStore 实例
    """
    if backend == "memory":
        return InMemoryStateStore()
    elif backend == "sqlite":
        db_path = kwargs.get("db_path")
        if not db_path:
            raise ValueError("db_path is required for sqlite backend")
        return SQLiteStateStore(db_path=db_path)
    elif backend == "redis_postgres":
        return RedisPostgresStateStore(
            redis_url=kwargs.get("redis_url", "redis://localhost:6379/0"),
            postgres_dsn=kwargs.get("postgres_dsn"),
            redis_prefix=kwargs.get("redis_prefix", "aurora:state:"),
            pg_table=kwargs.get("pg_table", "aurora_states"),
            redis_ttl=kwargs.get("redis_ttl", 3600 * 24),
        )
    else:
        raise ValueError(f"Unknown state store backend: {backend}")
