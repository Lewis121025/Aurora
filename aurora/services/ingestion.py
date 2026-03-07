"""
摄入服务 (写入路径)
==============================

处理 CQRS 的写入端:
- 验证传入的交互请求
- 发布到消息队列进行异步处理
- 立即返回确认

对于同步处理 (开发/低容量)，可以内联处理。
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
    """摄入新交互的请求。"""
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
    """排队摄入请求的确认。"""
    event_id: str
    status: str  # "queued", "processing", "completed", "failed"
    queued_at: float = field(default_factory=time.time)
    message: Optional[str] = None


class MessageQueue(Protocol):
    """消息队列后端的协议。"""

    async def publish(self, topic: str, message: str) -> None:
        """发布消息到主题。"""
        ...

    async def subscribe(self, topic: str) -> "MessageConsumer":
        """订阅主题。"""
        ...


class MessageConsumer(Protocol):
    """消息消费者的协议。"""

    async def receive(self) -> Optional[str]:
        """接收下一条消息，如果为空则返回 None。"""
        ...

    async def ack(self, message_id: str) -> None:
        """确认消息处理。"""
        ...


class InMemoryQueue:
    """用于开发/测试的内存消息队列。"""
    
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
    """InMemoryQueue 的内存消费者。"""
    
    def __init__(self, queue: asyncio.Queue):
        self._queue = queue
    
    async def receive(self, timeout: float = 1.0) -> Optional[str]:
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
    
    async def ack(self, message_id: str) -> None:
        # 内存队列不需要显式确认
        pass


class KafkaQueue:
    """Kafka 消息队列用于生产环境。

    需要安装 aiokafka。
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
                raise ImportError("aiokafka 是 KafkaQueue 必需的。安装方式: pip install aiokafka")
    
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
            raise ImportError("aiokafka 是 KafkaQueue 必需的。安装方式: pip install aiokafka")
    
    async def close(self):
        if self._producer is not None:
            await self._producer.stop()
            self._producer = None


class KafkaConsumer:
    """Kafka 消费者包装器。"""
    
    def __init__(self, consumer):
        self._consumer = consumer
    
    async def receive(self, timeout: float = 1.0) -> Optional[str]:
        try:
            msg = await asyncio.wait_for(self._consumer.getone(), timeout=timeout)
            return msg.value
        except asyncio.TimeoutError:
            return None
    
    async def ack(self, message_id: str) -> None:
        # Kafka 默认自动提交
        pass
    
    async def close(self):
        await self._consumer.stop()


class IngestionService:
    """CQRS 架构的写入服务。

    处理传入的交互请求:
    - 验证请求
    - 发布到消息队列
    - 立即返回确认

    对于开发/测试，可以使用 sync_mode=True 进行同步处理。
    """
    
    INGEST_TOPIC = "aurora.ingest"
    
    def __init__(
        self,
        queue: Optional[MessageQueue] = None,
        sync_mode: bool = False,
        sync_processor: Optional[Any] = None,
    ):
        """初始化摄入服务。

        参数:
            queue: 消息队列后端 (默认: InMemoryQueue)
            sync_mode: 如果为 True，不使用队列进行同步处理
            sync_processor: 在同步模式下使用的处理器 (例如: AuroraTenant)
        """
        self.queue = queue or InMemoryQueue()
        self.sync_mode = sync_mode
        self.sync_processor = sync_processor
        
        # Statistics
        self._total_requests = 0
        self._queued_count = 0
        self._sync_processed_count = 0
    
    async def ingest(self, request: IngestRequest) -> IngestAck:
        """摄入交互请求。

        在异步模式下: 发布到队列并立即返回。
        在同步模式下: 内联处理并返回结果。
        """
        self._total_requests += 1

        # 确保时间戳
        if request.ts is None:
            request.ts = time.time()

        if self.sync_mode and self.sync_processor is not None:
            # 同步处理 (开发模式)
            return await self._process_sync(request)
        else:
            # 通过队列进行异步处理
            return await self._enqueue(request)
    
    async def _enqueue(self, request: IngestRequest) -> IngestAck:
        """发布请求到消息队列。"""
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
        """同步处理请求 (用于开发/测试)。"""
        try:
            # 调用同步处理器 (例如: AuroraTenant.ingest_interaction)
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
        """返回服务统计信息。"""
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
    """创建消息队列的工厂函数。

    参数:
        backend: "memory" 或 "kafka"
        **kwargs: 后端特定的参数
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
