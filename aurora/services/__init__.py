"""
Aurora Services Layer (CQRS Architecture)
==========================================

Implements Command Query Responsibility Segregation:
- IngestionService: Write path - accepts interactions, queues for processing
- QueryService: Read path - direct queries against read-optimized stores
- IngestWorker: Background processor - consumes queue, processes with LLM

Design goals:
- Horizontal scaling: Multiple workers can process in parallel
- Non-blocking writes: Return ACK immediately, process asynchronously
- Read optimization: Queries bypass write path for low latency
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
