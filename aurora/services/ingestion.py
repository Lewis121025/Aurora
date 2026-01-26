"""
Ingestion Service (Write Path)
==============================

Handles the write side of CQRS:
- Validates incoming interaction requests
- Publishes to message queue for async processing
- Returns acknowledgment immediately

For synchronous processing (development/low-volume), can process inline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class IngestRequest(BaseModel):
    """Request to ingest a new interaction."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    session_id: str
    user_message: str
    agent_message: str
    actors: Optional[List[str]] = None
    context: Optional[str] = None
    ts: Optional[float] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "event_id": "evt_123",
                "user_id": "user_456",
                "session_id": "sess_789",
                "user_message": "How do I implement memory?",
                "agent_message": "You can use a narrative memory architecture...",
                "actors": ["user", "agent"],
                "context": "programming discussion",
            }
        }


@dataclass
class IngestAck:
    """Acknowledgment for queued ingest request."""
    event_id: str
    status: str  # "queued", "processing", "completed", "failed"
    queued_at: float = field(default_factory=time.time)
    message: Optional[str] = None


class MessageQueue(Protocol):
    """Protocol for message queue backends."""
    
    async def publish(self, topic: str, message: str) -> None:
        """Publish a message to a topic."""
        ...
    
    async def subscribe(self, topic: str) -> "MessageConsumer":
        """Subscribe to a topic."""
        ...


class MessageConsumer(Protocol):
    """Protocol for message consumers."""
    
    async def receive(self) -> Optional[str]:
        """Receive next message, or None if empty."""
        ...
    
    async def ack(self, message_id: str) -> None:
        """Acknowledge message processing."""
        ...


class InMemoryQueue:
    """In-memory message queue for development/testing."""
    
    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = {}
        self._lock = asyncio.Lock()
    
    async def publish(self, topic: str, message: str) -> None:
        async with self._lock:
            if topic not in self._queues:
                self._queues[topic] = asyncio.Queue()
        await self._queues[topic].put(message)
    
    async def subscribe(self, topic: str) -> "InMemoryConsumer":
        async with self._lock:
            if topic not in self._queues:
                self._queues[topic] = asyncio.Queue()
        return InMemoryConsumer(self._queues[topic])


class InMemoryConsumer:
    """In-memory consumer for InMemoryQueue."""
    
    def __init__(self, queue: asyncio.Queue):
        self._queue = queue
    
    async def receive(self, timeout: float = 1.0) -> Optional[str]:
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
    
    async def ack(self, message_id: str) -> None:
        # In-memory queue doesn't need explicit acks
        pass


class KafkaQueue:
    """Kafka message queue for production.
    
    Requires aiokafka to be installed.
    """
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        client_id: str = "aurora-ingestion",
    ):
        self.bootstrap_servers = bootstrap_servers
        self.client_id = client_id
        self._producer = None
    
    async def _ensure_producer(self):
        if self._producer is None:
            try:
                from aiokafka import AIOKafkaProducer
                self._producer = AIOKafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    client_id=self.client_id,
                    value_serializer=lambda v: v.encode("utf-8"),
                )
                await self._producer.start()
            except ImportError:
                raise ImportError("aiokafka is required for KafkaQueue. Install with: pip install aiokafka")
    
    async def publish(self, topic: str, message: str) -> None:
        await self._ensure_producer()
        await self._producer.send_and_wait(topic, message)
    
    async def subscribe(self, topic: str) -> "KafkaConsumer":
        try:
            from aiokafka import AIOKafkaConsumer
            consumer = AIOKafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=f"{self.client_id}-group",
                auto_offset_reset="earliest",
                value_deserializer=lambda v: v.decode("utf-8"),
            )
            await consumer.start()
            return KafkaConsumer(consumer)
        except ImportError:
            raise ImportError("aiokafka is required for KafkaQueue. Install with: pip install aiokafka")
    
    async def close(self):
        if self._producer is not None:
            await self._producer.stop()
            self._producer = None


class KafkaConsumer:
    """Kafka consumer wrapper."""
    
    def __init__(self, consumer):
        self._consumer = consumer
    
    async def receive(self, timeout: float = 1.0) -> Optional[str]:
        try:
            msg = await asyncio.wait_for(self._consumer.getone(), timeout=timeout)
            return msg.value
        except asyncio.TimeoutError:
            return None
    
    async def ack(self, message_id: str) -> None:
        # Kafka auto-commits by default
        pass
    
    async def close(self):
        await self._consumer.stop()


class IngestionService:
    """Write service for CQRS architecture.
    
    Handles incoming interaction requests:
    - Validates the request
    - Publishes to message queue
    - Returns acknowledgment immediately
    
    For development/testing, can process synchronously with sync_mode=True.
    """
    
    INGEST_TOPIC = "aurora.ingest"
    
    def __init__(
        self,
        queue: Optional[MessageQueue] = None,
        sync_mode: bool = False,
        sync_processor: Optional[Any] = None,
    ):
        """Initialize the ingestion service.
        
        Args:
            queue: Message queue backend (default: InMemoryQueue)
            sync_mode: If True, process synchronously without queue
            sync_processor: Processor to use in sync mode (e.g., AuroraTenant)
        """
        self.queue = queue or InMemoryQueue()
        self.sync_mode = sync_mode
        self.sync_processor = sync_processor
        
        # Statistics
        self._total_requests = 0
        self._queued_count = 0
        self._sync_processed_count = 0
    
    async def ingest(self, request: IngestRequest) -> IngestAck:
        """Ingest an interaction request.
        
        In async mode: Publishes to queue and returns immediately.
        In sync mode: Processes inline and returns result.
        """
        self._total_requests += 1
        
        # Ensure timestamp
        if request.ts is None:
            request.ts = time.time()
        
        if self.sync_mode and self.sync_processor is not None:
            # Synchronous processing (development mode)
            return await self._process_sync(request)
        else:
            # Async processing via queue
            return await self._enqueue(request)
    
    async def _enqueue(self, request: IngestRequest) -> IngestAck:
        """Publish request to message queue."""
        try:
            message = request.model_dump_json()
            await self.queue.publish(self.INGEST_TOPIC, message)
            self._queued_count += 1
            
            return IngestAck(
                event_id=request.event_id,
                status="queued",
                message="Request queued for processing",
            )
        except Exception as e:
            logger.error(f"Failed to queue request {request.event_id}: {e}")
            return IngestAck(
                event_id=request.event_id,
                status="failed",
                message=f"Failed to queue: {str(e)}",
            )
    
    async def _process_sync(self, request: IngestRequest) -> IngestAck:
        """Process request synchronously (for development/testing)."""
        try:
            # Call the sync processor (e.g., AuroraTenant.ingest_interaction)
            result = self.sync_processor.ingest_interaction(
                event_id=request.event_id,
                session_id=request.session_id,
                user_message=request.user_message,
                agent_message=request.agent_message,
                actors=request.actors,
                context=request.context,
                ts=request.ts,
            )
            self._sync_processed_count += 1
            
            return IngestAck(
                event_id=request.event_id,
                status="completed",
                message=f"Processed synchronously, plot_id={result.plot_id}",
            )
        except Exception as e:
            logger.error(f"Sync processing failed for {request.event_id}: {e}")
            return IngestAck(
                event_id=request.event_id,
                status="failed",
                message=f"Processing failed: {str(e)}",
            )
    
    def stats(self) -> Dict[str, Any]:
        """Return service statistics."""
        return {
            "total_requests": self._total_requests,
            "queued_count": self._queued_count,
            "sync_processed_count": self._sync_processed_count,
            "sync_mode": self.sync_mode,
        }


def create_queue(
    backend: str = "memory",
    **kwargs,
) -> MessageQueue:
    """Factory function to create message queue.
    
    Args:
        backend: "memory" or "kafka"
        **kwargs: Backend-specific arguments
    """
    if backend == "memory":
        return InMemoryQueue()
    elif backend == "kafka":
        return KafkaQueue(
            bootstrap_servers=kwargs.get("bootstrap_servers", "localhost:9092"),
            client_id=kwargs.get("client_id", "aurora-ingestion"),
        )
    else:
        raise ValueError(f"Unknown queue backend: {backend}")
