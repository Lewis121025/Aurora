"""冷事实账本模块。

实现客观账本（ObjectiveLedger）：
- 原子事实存储
- 384维向量相似度查询（MiniLM）
- 时间衰减权重
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class AtomicFact:
    """原子事实。

    Attributes:
        fact_id: 事实 ID。
        content: 事实内容。
        document_date: 记录时间戳。
        event_date: 事件时间戳。
        relation_id: 关系 ID。
    """

    fact_id: str
    content: str
    document_date: float
    event_date: float
    relation_id: str


EMBEDDING_DIM = 384
"""向量维度（MiniLM）。"""


class ObjectiveLedger:
    """客观账本。

    SQLite + 内存向量存储，管理原子事实的持久化和检索。

    Attributes:
        conn: SQLite 连接。
        _embeddings: 内存向量缓存。
    """

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_path = ".aurora/ledger.db"
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._embeddings: dict[str, np.ndarray] = {}
        self._init_schema()

    def _init_schema(self) -> None:
        """初始化数据库 schema。"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                fact_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                document_date REAL NOT NULL,
                event_date REAL NOT NULL,
                relation_id TEXT NOT NULL,
                embedding BLOB
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_relation ON facts(relation_id)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_date ON facts(event_date)
        """)
        self.conn.commit()

    def add_fact(
        self,
        fact_id: str,
        content: str,
        document_date: float,
        event_date: float,
        relation_id: str,
        embedding: np.ndarray | None = None,
    ) -> None:
        """添加原子事实。

        Args:
            fact_id: 事实 ID。
            content: 事实内容。
            document_date: 记录时间戳。
            event_date: 事件时间戳。
            relation_id: 关系 ID。
            embedding: 向量嵌入（可选）。
        """
        emb_bytes = embedding.tobytes() if embedding is not None else None
        self.conn.execute(
            "INSERT OR REPLACE INTO facts (fact_id, content, document_date, event_date, relation_id, embedding) VALUES (?, ?, ?, ?, ?, ?)",
            (fact_id, content, document_date, event_date, relation_id, emb_bytes),
        )
        self.conn.commit()
        if embedding is not None:
            self._embeddings[fact_id] = embedding

    def query_by_similarity(
        self,
        query_embedding: np.ndarray,
        relation_id: str | None = None,
        top_k: int = 5,
    ) -> list[AtomicFact]:
        """按向量相似度查询。

        Args:
            query_embedding: 查询向量。
            relation_id: 关系 ID 过滤（可选）。
            top_k: 返回数量。

        Returns:
            原子事实列表（按相似度降序）。
        """
        if relation_id:
            rows = self.conn.execute(
                "SELECT fact_id, content, document_date, event_date, relation_id, embedding FROM facts WHERE relation_id = ?",
                (relation_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT fact_id, content, document_date, event_date, relation_id, embedding FROM facts",
            ).fetchall()

        results: list[tuple[float, AtomicFact]] = []
        for row in rows:
            fact_id, content, doc_date, evt_date, rel_id, emb_bytes = row
            if emb_bytes is None:
                continue
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            emb = emb / np.linalg.norm(emb)
            sim = float(np.dot(query_embedding, emb))
            results.append((sim, AtomicFact(fact_id, content, doc_date, evt_date, rel_id)))

        results.sort(key=lambda x: x[0], reverse=True)
        return [fact for _, fact in results[:top_k]]

    def facts_for_relation(self, relation_id: str) -> list[AtomicFact]:
        """获取关系的所有事实。

        Args:
            relation_id: 关系 ID。

        Returns:
            原子事实列表。
        """
        rows = self.conn.execute(
            "SELECT fact_id, content, document_date, event_date, relation_id FROM facts WHERE relation_id = ? ORDER BY event_date DESC",
            (relation_id,),
        ).fetchall()
        return [AtomicFact(*row) for row in rows]

    def close(self) -> None:
        """关闭连接。"""
        self.conn.close()
