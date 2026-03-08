"""
摄入 Worker (后台处理)
=====================================

从摄入队列消费消息并处理它们:
1. LLM 情节提取
2. 嵌入计算
3. 向量存储更新
4. 状态存储更新
5. 图编织

支持多个 worker 的水平扩展。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ProcessResult:
    """摄入请求的处理结果。"""
    event_id: str
    plot_id: str
    story_id: Optional[str]
    encoded: bool
    tension: float
    surprise: float
    pred_error: float
    redundancy: float
    processing_time_ms: float
    error: Optional[str] = None


class IngestWorker:
    """后台 worker 用于处理摄入请求。

    从消息队列消费并处理:
    1. 使用 LLM 从交互中提取情节
    2. 计算嵌入
    3. 应用内存算法 (gate、CRP 等)
    4. 写入向量存储和状态存储

    支持水平扩展 - 多个 worker 可以从同一队列消费。
    """
    
    def __init__(
        self,
        queue,
        topic: str = "aurora.ingest",
        llm_provider=None,
        embedding_provider=None,
        vector_store=None,
        state_store=None,
        memory=None,  # AuroraMemory for algorithm logic
        batch_size: int = 10,
        poll_interval: float = 0.1,
    ):
        """初始化 worker。

        参数:
            queue: 要消费的消息队列
            topic: 要订阅的主题
            llm_provider: 用于情节提取的 LLM
            embedding_provider: 嵌入提供者
            vector_store: 用于写入的向量存储
            state_store: 用于算法状态的状态存储
            memory: AuroraMemory 实例 (可选，用于同步处理)
            batch_size: 批量处理的消息数
            poll_interval: 队列轮询之间的秒数
        """
        self.queue = queue
        self.topic = topic
        self.llm = llm_provider
        self.embedder = embedding_provider
        self.vector_store = vector_store
        self.state_store = state_store
        self.memory = memory
        self.batch_size = batch_size
        self.poll_interval = poll_interval

        self._running = False
        self._consumer = None

        # 统计信息
        self._processed_count = 0
        self._error_count = 0
        self._total_processing_time = 0.0
    
    async def start(self) -> None:
        """启动 worker。"""
        logger.info(f"Starting IngestWorker for topic: {self.topic}")
        self._running = True
        self._consumer = await self.queue.subscribe(self.topic)

        while self._running:
            try:
                await self._process_batch()
            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(1.0)  # 出错时退避
    
    async def stop(self) -> None:
        """优雅地停止 worker。"""
        logger.info("Stopping IngestWorker")
        self._running = False
        if hasattr(self._consumer, "close"):
            await self._consumer.close()
    
    async def _process_batch(self) -> int:
        """处理一批消息。返回处理的数量。"""
        processed = 0

        for _ in range(self.batch_size):
            if not self._running:
                break

            # 接收消息
            msg = await self._consumer.receive(timeout=self.poll_interval)
            if msg is None:
                break

            # 处理消息
            try:
                result = await self._process_message(msg)
                await self._consumer.ack(result.event_id)
                self._processed_count += 1
                processed += 1
            except Exception as e:
                logger.error(f"Failed to process message: {e}")
                self._error_count += 1

        return processed
    
    async def _process_message(self, msg: str) -> ProcessResult:
        """处理单个消息。"""
        start_time = time.time()

        # 解析请求
        from aurora.runtime.cqrs.ingestion import IngestRequest
        request = IngestRequest.model_validate_json(msg)

        # 使用内存实例处理 (如果可用)
        if self.memory is not None:
            result = await self._process_with_memory(request)
        else:
            result = await self._process_distributed(request)

        result.processing_time_ms = (time.time() - start_time) * 1000
        self._total_processing_time += result.processing_time_ms

        return result
    
    async def _process_with_memory(self, request) -> ProcessResult:
        """使用内存中的 AuroraMemory 处理。"""
        from aurora.runtime.cqrs.ingestion import IngestRequest

        def _sync_process():
            # 构建交互文本
            if self.llm is not None:
                # 使用 LLM 进行提取
                from aurora.integrations.llm import prompts
                from aurora.integrations.llm.schemas import PlotExtraction

                instruction = prompts.instruction("PlotExtraction")
                user_prompt = prompts.render(
                    prompts.PLOT_EXTRACTION_USER,
                    instruction=instruction,
                    user_message=request.user_message,
                    agent_message=request.agent_message,
                    context=request.context or "",
                )

                try:
                    extraction = self.llm.complete_json(
                        system=prompts.PLOT_EXTRACTION_SYSTEM,
                        user=user_prompt,
                        schema=PlotExtraction,
                        temperature=0.2,
                        timeout_s=20.0,
                    )
                    outcome = extraction.outcome
                    actors = extraction.actors
                except Exception as e:
                    logger.debug(f"LLM plot extraction failed, using defaults: {e}")
                    outcome = ""
                    actors = request.actors or ["user", "agent"]
            else:
                outcome = ""
                actors = request.actors or ["user", "agent"]

            interaction_text = f"USER: {request.user_message}\nAGENT: {request.agent_message}"
            if outcome:
                interaction_text += f"\nOUTCOME: {outcome}"

            # 摄入到内存
            plot = self.memory.ingest(
                interaction_text,
                actors=actors,
                context_text=request.context,
                event_id=request.event_id,
            )

            encoded = plot.id in self.memory.plots

            return ProcessResult(
                event_id=request.event_id,
                plot_id=plot.id,
                story_id=plot.story_id,
                encoded=encoded,
                tension=float(plot.tension),
                surprise=float(plot.surprise),
                pred_error=float(plot.pred_error),
                redundancy=float(plot.redundancy),
                processing_time_ms=0.0,
            )

        # 在线程池中运行以不阻塞事件循环
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_process)
    
    async def _process_distributed(self, request) -> ProcessResult:
        """在分布式模式下处理 (向量存储 + 状态存储)。"""
        # 这是完全分布式处理的占位符
        # 其中每个组件通过网络访问

        # 目前，返回一个错误，指示需要内存
        return ProcessResult(
            event_id=request.event_id,
            plot_id="",
            story_id=None,
            encoded=False,
            tension=0.0,
            surprise=0.0,
            pred_error=0.0,
            redundancy=0.0,
            processing_time_ms=0.0,
            error="Distributed processing not yet implemented, use memory parameter",
        )
    
    def stats(self) -> Dict[str, Any]:
        """返回 worker 统计信息。"""
        avg_time = self._total_processing_time / max(1, self._processed_count)
        return {
            "processed_count": self._processed_count,
            "error_count": self._error_count,
            "avg_processing_time_ms": avg_time,
            "running": self._running,
        }


class WorkerPool:
    """用于水平扩展的 worker 池。"""
    
    def __init__(
        self,
        num_workers: int = 4,
        worker_factory=None,
    ):
        """初始化 worker 池。

        参数:
            num_workers: 要运行的 worker 数量
            worker_factory: 返回 IngestWorker 实例的可调用对象
        """
        self.num_workers = num_workers
        self.worker_factory = worker_factory
        self._workers: List[IngestWorker] = []
        self._tasks: List[asyncio.Task] = []
    
    async def start(self) -> None:
        """启动所有 worker。"""
        logger.info(f"Starting worker pool with {self.num_workers} workers")

        for i in range(self.num_workers):
            worker = self.worker_factory()
            self._workers.append(worker)
            task = asyncio.create_task(worker.start())
            self._tasks.append(task)
    
    async def stop(self) -> None:
        """优雅地停止所有 worker。"""
        logger.info("Stopping worker pool")

        # 向所有 worker 发送停止信号
        for worker in self._workers:
            await worker.stop()

        # 等待任务完成
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        self._workers.clear()
        self._tasks.clear()
    
    def stats(self) -> Dict[str, Any]:
        """从所有 worker 聚合统计信息。"""
        total_processed = sum(w._processed_count for w in self._workers)
        total_errors = sum(w._error_count for w in self._workers)
        total_time = sum(w._total_processing_time for w in self._workers)
        
        return {
            "num_workers": len(self._workers),
            "total_processed": total_processed,
            "total_errors": total_errors,
            "avg_processing_time_ms": total_time / max(1, total_processed),
        }
