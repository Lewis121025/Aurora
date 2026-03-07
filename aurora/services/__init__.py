"""
Aurora Services Layer (CQRS Architecture)
==========================================

实现命令查询职责分离 (CQRS):
- IngestionService: 写入路径 - 接受交互，排队处理
- QueryService: 读取路径 - 直接查询读优化存储
- IngestWorker: 后台处理器 - 消费队列，使用 LLM 处理

设计目标:
- 水平扩展: 多个 worker 可以并行处理
- 非阻塞写入: 立即返回 ACK，异步处理
- 读取优化: 查询绕过写入路径以实现低延迟
"""

from aurora.services.ingestion import (
    IngestionService,
    IngestAck,
    IngestRequest,
    InMemoryQueue,
    KafkaQueue,
    create_queue,
)
from aurora.services.query import (
    QueryService,
    QueryRequest,
    QueryResponse,
    QueryHit,
    QueryCache,
)
from aurora.services.worker import (
    IngestWorker,
    ProcessResult,
    WorkerPool,
)

__all__ = [
    # Ingestion
    "IngestionService",
    "IngestAck",
    "IngestRequest",
    "InMemoryQueue",
    "KafkaQueue",
    "create_queue",
    # Query
    "QueryService",
    "QueryRequest",
    "QueryResponse",
    "QueryHit",
    "QueryCache",
    # Worker
    "IngestWorker",
    "ProcessResult",
    "WorkerPool",
]
