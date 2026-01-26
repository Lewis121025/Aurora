"""
Ingest Worker (Background Processing)
=====================================

Consumes messages from the ingestion queue and processes them:
1. LLM Plot extraction
2. Embedding computation
3. Vector store updates
4. State store updates
5. Graph weaving

Supports horizontal scaling with multiple workers.
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
    """Result of processing an ingest request."""
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
    """Background worker for processing ingest requests.
    
    Consumes from message queue and processes:
    1. Extract plot from interaction using LLM
    2. Compute embedding
    3. Apply memory algorithm (gate, CRP, etc.)
    4. Write to vector store and state store
    
    Supports horizontal scaling - multiple workers can consume from the same queue.
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
        """Initialize the worker.
        
        Args:
            queue: Message queue to consume from
            topic: Topic to subscribe to
            llm_provider: LLM for plot extraction
            embedding_provider: Embedding provider
            vector_store: Vector store for writes
            state_store: State store for algorithm state
            memory: AuroraMemory instance (optional, for sync processing)
            batch_size: Number of messages to process in batch
            poll_interval: Seconds between queue polls
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
        
        # Statistics
        self._processed_count = 0
        self._error_count = 0
        self._total_processing_time = 0.0
    
    async def start(self) -> None:
        """Start the worker."""
        logger.info(f"Starting IngestWorker for topic: {self.topic}")
        self._running = True
        self._consumer = await self.queue.subscribe(self.topic)
        
        while self._running:
            try:
                await self._process_batch()
            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(1.0)  # Back off on error
    
    async def stop(self) -> None:
        """Stop the worker gracefully."""
        logger.info("Stopping IngestWorker")
        self._running = False
        if hasattr(self._consumer, "close"):
            await self._consumer.close()
    
    async def _process_batch(self) -> int:
        """Process a batch of messages. Returns count processed."""
        processed = 0
        
        for _ in range(self.batch_size):
            if not self._running:
                break
            
            # Receive message
            msg = await self._consumer.receive(timeout=self.poll_interval)
            if msg is None:
                break
            
            # Process message
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
        """Process a single message."""
        start_time = time.time()
        
        # Parse request
        from aurora.services.ingestion import IngestRequest
        request = IngestRequest.model_validate_json(msg)
        
        # Process using memory instance if available
        if self.memory is not None:
            result = await self._process_with_memory(request)
        else:
            result = await self._process_distributed(request)
        
        result.processing_time_ms = (time.time() - start_time) * 1000
        self._total_processing_time += result.processing_time_ms
        
        return result
    
    async def _process_with_memory(self, request) -> ProcessResult:
        """Process using in-memory AuroraMemory."""
        from aurora.services.ingestion import IngestRequest
        
        def _sync_process():
            # Build interaction text
            if self.llm is not None:
                # Use LLM for extraction
                from aurora.llm import prompts
                from aurora.llm.schemas import PlotExtraction
                
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
                except Exception:
                    outcome = ""
                    actors = request.actors or ["user", "agent"]
            else:
                outcome = ""
                actors = request.actors or ["user", "agent"]
            
            interaction_text = f"USER: {request.user_message}\nAGENT: {request.agent_message}"
            if outcome:
                interaction_text += f"\nOUTCOME: {outcome}"
            
            # Ingest into memory
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
        
        # Run in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_process)
    
    async def _process_distributed(self, request) -> ProcessResult:
        """Process in distributed mode (vector store + state store)."""
        # This is a placeholder for fully distributed processing
        # where each component is accessed via network
        
        # For now, return an error indicating memory is required
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
        """Return worker statistics."""
        avg_time = self._total_processing_time / max(1, self._processed_count)
        return {
            "processed_count": self._processed_count,
            "error_count": self._error_count,
            "avg_processing_time_ms": avg_time,
            "running": self._running,
        }


class WorkerPool:
    """Pool of workers for horizontal scaling."""
    
    def __init__(
        self,
        num_workers: int = 4,
        worker_factory=None,
    ):
        """Initialize worker pool.
        
        Args:
            num_workers: Number of workers to run
            worker_factory: Callable that returns IngestWorker instances
        """
        self.num_workers = num_workers
        self.worker_factory = worker_factory
        self._workers: List[IngestWorker] = []
        self._tasks: List[asyncio.Task] = []
    
    async def start(self) -> None:
        """Start all workers."""
        logger.info(f"Starting worker pool with {self.num_workers} workers")
        
        for i in range(self.num_workers):
            worker = self.worker_factory()
            self._workers.append(worker)
            task = asyncio.create_task(worker.start())
            self._tasks.append(task)
    
    async def stop(self) -> None:
        """Stop all workers gracefully."""
        logger.info("Stopping worker pool")
        
        # Signal all workers to stop
        for worker in self._workers:
            await worker.stop()
        
        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        self._workers.clear()
        self._tasks.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Aggregate statistics from all workers."""
        total_processed = sum(w._processed_count for w in self._workers)
        total_errors = sum(w._error_count for w in self._workers)
        total_time = sum(w._total_processing_time for w in self._workers)
        
        return {
            "num_workers": len(self._workers),
            "total_processed": total_processed,
            "total_errors": total_errors,
            "avg_processing_time_ms": total_time / max(1, total_processed),
        }
