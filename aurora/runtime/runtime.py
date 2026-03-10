"""
aurora/runtime/runtime.py
运行时核心模块：负责整合底层灵魂引擎、持久化存储、大模型接口以及响应构建逻辑。
它管理着 Agent 的生命周期，包括启动加载、交互写回、梦境演变以及检索。
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional, Sequence

from aurora.system.errors import ConfigurationError
from aurora.integrations.llm.provider import LLMProvider
from aurora.integrations.storage.doc_store import Document, SQLiteDocStore
from aurora.integrations.storage.event_log import Event, SQLiteEventLog
from aurora.integrations.storage.snapshot import Snapshot, SnapshotStore
from aurora.runtime.bootstrap import (
    check_embedding_api_keys,
    create_embedding_provider,
    create_llm_provider,
    create_meaning_provider,
    create_narrative_provider,
    create_memory,
)
from aurora.runtime.response_context import ResponseContextBuilder
from aurora.runtime.results import (
    ChatStreamEvent,
    ChatTimings,
    ChatTurnResult,
    IngestResult,
    QueryHit,
    QueryResult,
    StructuredMemoryContext,
)
from aurora.runtime.settings import AuroraSettings, DEFAULT_DATA_DIR
from aurora.soul.engine import AuroraSoul

logger = logging.getLogger(__name__)

# 运行时架构版本标识
RUNTIME_SCHEMA_VERSION = "aurora-runtime-v4"


class AuroraRuntime:
    """
    Aurora 运行时：这是整个系统的中枢。
    它不仅持有 AuroraSoul 引擎，还负责与数据库（SQLite）、大模型 API 的具体通信。
    """
    def __init__(self, *, settings: AuroraSettings, llm: Optional[LLMProvider] = None):
        self.settings = settings
        self.llm = llm if llm is not None else create_llm_provider(settings)

        # 启动前自检
        check_embedding_api_keys()
        self._lock = threading.RLock() # 确保多线程下的内存状态安全

        # 初始化持久化存储
        os.makedirs(self.settings.data_dir, exist_ok=True)
        # 1. 事件日志：记录原始交互流（Source of Truth）
        self.event_log = SQLiteEventLog(os.path.join(self.settings.data_dir, self.settings.event_log_filename))
        # 2. 文档存储：存储解析后的情节、结果快照等
        self.doc_store = SQLiteDocStore(os.path.join(self.settings.data_dir, "docs.sqlite3"))
        # 3. 快照存储：存储序列化后的引擎完整状态
        self.snapshots = SnapshotStore(os.path.join(self.settings.data_dir, self.settings.snapshot_dirname))

        self.last_seq = 0
        self._loaded_snapshot_seq = 0
        # 从快照加载引擎，若无则初始化
        self.mem = self._load_or_init()
        self.response_contexts = ResponseContextBuilder(memory=self.mem)

        # 检查版本兼容性并重放未同步的日志
        self._ensure_v4_doc_store()
        replayed = self._replay()
        if replayed > 0 and self.last_seq > self._loaded_snapshot_seq:
            self._snapshot()

    def _load_or_init(self) -> AuroraSoul:
        """从最新快照恢复系统状态，或根据配置新建引擎。"""
        try:
            latest = self.snapshots.latest()
        except ValueError as exc:
            raise ConfigurationError(str(exc).replace("V2", "Aurora Soul V4")) from exc
            
        if latest is not None:
            _seq, snap = latest
            self.last_seq = snap.last_seq
            self._loaded_snapshot_seq = snap.last_seq
            # 恢复时重新配置 Provider
            event_embedder = create_embedding_provider(self.settings)
            axis_embedder = create_embedding_provider(
                self.settings,
                provider_override=self.settings.axis_embedding_provider or self.settings.embedding_provider,
            )
            meaning_provider = create_meaning_provider(settings=self.settings, llm=self.llm)
            narrator = create_narrative_provider(settings=self.settings, llm=self.llm)
            try:
                return AuroraSoul.from_state_dict(
                    snap.state,
                    event_embedder=event_embedder,
                    axis_embedder=axis_embedder,
                    meaning_provider=meaning_provider,
                    narrator=narrator,
                )
            except ValueError as exc:
                raise ConfigurationError(str(exc)) from exc
        return create_memory(settings=self.settings, llm=self.llm)

    def _ensure_v4_doc_store(self) -> None:
        """强制检查数据一致性，防止 v3 版本的数据污染 v4 系统。"""
        for kind in ("plot", "ingest_result"):
            if self.doc_store.has_body_field_mismatch(
                kind=kind,
                field="runtime_schema_version",
                expected=RUNTIME_SCHEMA_VERSION,
            ):
                raise ConfigurationError(
                    f"Detected legacy v3 documents in {self.settings.data_dir}/docs.sqlite3. "
                    f"Aurora Soul v4 requires a fresh data directory, for example {DEFAULT_DATA_DIR!r}."
                )

    def _ensure_v4_event_payload(self, payload: Dict[str, Any]) -> None:
        """检查单条事件的架构版本。"""
        if payload.get("runtime_schema_version") != RUNTIME_SCHEMA_VERSION:
            raise ConfigurationError(
                f"Detected legacy v3 event payloads in {self.settings.data_dir}. "
                "Aurora Soul v4 requires a fresh data directory."
            )

    def _replay(self) -> int:
        """重放日志逻辑：将快照之后产生的事件重新喂给内存引擎，保持内存与日志同步。"""
        replayed = 0
        for seq, event in self.event_log.iter_events(after_seq=self.last_seq):
            if event.type != "interaction":
                continue
            self._ensure_v4_event_payload(event.payload)
            self._apply_interaction(
                event_id=event.id,
                user_message=str(event.payload.get("user_message", "")),
                agent_message=str(event.payload.get("agent_message", "")),
                context=event.payload.get("context"),
                actors=event.payload.get("actors"),
                ts=event.ts,
                persist=False, # 重放时不重复持久化
            )
            self.last_seq = seq
            replayed += 1
        return replayed

    def ingest_interaction(
        self,
        *,
        event_id: str,
        session_id: str,
        user_message: str,
        agent_message: str,
        actors: Optional[Sequence[str]] = None,
        context: Optional[str] = None,
        ts: Optional[float] = None,
    ) -> IngestResult:
        """
        交互摄入流程：
        1. 检查去重。
        2. 写入事件日志。
        3. 喂给灵魂引擎。
        4. 定期触发系统快照。
        """
        event_ts = ts or time.time()
        with self._lock:
            existing_seq = self.event_log.get_seq_by_id(event_id)
            if existing_seq is not None:
                return self._load_existing_ingest_result(event_id)

            payload = self._build_event_payload(
                user_message=user_message,
                agent_message=agent_message,
                actors=actors,
                context=context,
            )
            seq = self.event_log.append(
                Event(
                    id=event_id,
                    ts=event_ts,
                    session_id=session_id,
                    type="interaction",
                    payload=payload,
                )
            )
            result = self._apply_interaction(
                event_id=event_id,
                user_message=user_message,
                agent_message=agent_message,
                context=context,
                actors=actors,
                ts=event_ts,
                persist=True,
            )
            self.last_seq = max(self.last_seq, seq)
            if self.settings.snapshot_every_events > 0 and self.last_seq % self.settings.snapshot_every_events == 0:
                self._snapshot()
            return result

    def _build_event_payload(
        self,
        *,
        user_message: str,
        agent_message: str,
        actors: Optional[Sequence[str]],
        context: Optional[str],
    ) -> Dict[str, Any]:
        """构建事件载荷字典。"""
        return {
            "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
            "user_message": user_message,
            "agent_message": agent_message,
            "actors": list(actors) if actors else ["user", "agent"],
            "context": context,
        }

    def _interaction_text(self, *, user_message: str, agent_message: str) -> str:
        """将对话双方消息合并为单段文本，供引擎分析含义。"""
        return f"USER: {user_message}\nAURORA: {agent_message}"

    def _apply_interaction(
        self,
        *,
        event_id: str,
        user_message: str,
        agent_message: str,
        context: Optional[str],
        actors: Optional[Sequence[str]],
        ts: float,
        persist: bool,
    ) -> IngestResult:
        """内部逻辑：调用 soul 引擎执行真正的摄入与身份计算。"""
        plot = self.mem.ingest(
            self._interaction_text(user_message=user_message, agent_message=agent_message),
            actors=actors or ("user", "agent"),
            context_text=context or user_message,
            source="wake",
            ts=ts,
        )
        result = IngestResult(
            event_id=event_id,
            plot_id=plot.id,
            story_id=plot.story_id,
            mode=self.mem.identity.current_mode_label,
            source=plot.source,
            tension=float(plot.tension),
            contradiction=float(plot.contradiction),
            active_energy=float(self.mem.identity.active_energy),
            repressed_energy=float(self.mem.identity.repressed_energy),
        )
        if persist:
            self._persist_interaction_artifacts(event_id=event_id, ts=ts, plot=plot, result=result)
        return result

    def _persist_interaction_artifacts(self, *, event_id: str, ts: float, plot: Any, result: IngestResult) -> None:
        """将摄入产物（情节和结果）保存到文档存储中。"""
        self.doc_store.upsert(
            Document(
                id=plot.id,
                kind="plot",
                ts=ts,
                body={
                    "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
                    "plot_id": plot.id,
                    "story_id": plot.story_id,
                    "text": plot.text,
                    "source": plot.source,
                    "mode": self.mem.identity.current_mode_label,
                    "tags": list(plot.frame.tags),
                    "tension": float(plot.tension),
                    "contradiction": float(plot.contradiction),
                },
            )
        )
        self.doc_store.upsert(
            Document(
                id=f"ingest:{event_id}",
                kind="ingest_result",
                ts=ts,
                body={
                    "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
                    "event_id": result.event_id,
                    "plot_id": result.plot_id,
                    "story_id": result.story_id,
                    "mode": result.mode,
                    "source": result.source,
                    "tension": result.tension,
                    "contradiction": result.contradiction,
                    "active_energy": result.active_energy,
                    "repressed_energy": result.repressed_energy,
                },
            )
        )

    def _snapshot(self) -> None:
        """执行全量系统快照。"""
        snap = Snapshot(last_seq=self.last_seq, state=self.mem.to_state_dict())
        self.snapshots.save(snap)

    def _load_existing_ingest_result(self, event_id: str) -> IngestResult:
        """从文档库加载已有的摄入结果。"""
        doc = self.doc_store.get(f"ingest:{event_id}")
        if doc is None:
            return IngestResult(
                event_id=event_id,
                plot_id="",
                story_id=None,
                mode=self.mem.identity.current_mode_label,
                source="wake",
                tension=0.0,
                contradiction=0.0,
                active_energy=float(self.mem.identity.active_energy),
                repressed_energy=float(self.mem.identity.repressed_energy),
            )
        body = doc.body
        return IngestResult(
            event_id=event_id,
            plot_id=str(body.get("plot_id", "")),
            story_id=body.get("story_id"),
            mode=str(body.get("mode", self.mem.identity.current_mode_label)),
            source=str(body.get("source", "wake")),
            tension=float(body.get("tension", 0.0)),
            contradiction=float(body.get("contradiction", 0.0)),
            active_energy=float(body.get("active_energy", self.mem.identity.active_energy)),
            repressed_energy=float(body.get("repressed_energy", self.mem.identity.repressed_energy)),
        )

    def build_response_context(self, *, user_message: str, k: int = 6) -> tuple[StructuredMemoryContext, Any]:
        """检索并构建当前对话所需的记忆上下文。"""
        with self._lock:
            trace = self.mem.query(user_message, k=k)
            context = self.response_contexts.build(trace=trace, max_items=k)
            summary = self.response_contexts.summarize_trace(trace)
        return context, summary

    def _response_llm_timeout_s(self) -> float:
        """交互式回复链路采用更短超时，避免终端长时间挂起。"""
        return min(float(self.settings.llm_timeout), 10.0)

    def _response_llm_max_retries(self) -> int:
        """回复链路默认快速失败，必要时由 runtime 自身生成降级回复。"""
        return 1

    def respond_stream(
        self,
        *,
        session_id: str,
        user_message: str,
        event_id: Optional[str] = None,
        context: Optional[str] = None,
        actors: Optional[Sequence[str]] = None,
        k: int = 6,
        ts: Optional[float] = None,
    ) -> Iterator[ChatStreamEvent]:
        """
        流式对话主流程：
        1. 检索：获取记忆上下文。
        2. 生成：流式调用 LLM 生成回复。
        3. 摄入：将用户输入和生成回复作为新的情节存回记忆。
        """
        total_started = time.perf_counter()
        resolved_event_id = event_id or f"evt_resp_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

        yield ChatStreamEvent(kind="status", stage="retrieval", text="正在检索记忆")
        retrieval_started = time.perf_counter()
        memory_context, trace_summary = self.build_response_context(user_message=user_message, k=k)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0

        prompt = self.response_contexts.build_prompt(
            user_message=user_message,
            rendered_memory_brief=self.response_contexts.render_memory_brief(memory_context),
        )

        yield ChatStreamEvent(kind="status", stage="generation", text="正在生成回复")
        generation_started = time.perf_counter()
        llm_error: Optional[str] = None
        reply_parts: List[str] = []
        if self.llm is None:
            llm_error = "llm_not_configured"
            reply = self._build_response_fallback(
                memory_context=memory_context,
                user_message=user_message,
                reason=llm_error,
            )
            yield ChatStreamEvent(kind="status", stage="generation", text="未配置语言模型，使用系统降级回复")
            yield ChatStreamEvent(kind="reply_delta", stage="generation", text=reply)
        else:
            try:
                # 调用流式 LLM 提供者
                for chunk in self.llm.stream_complete(
                    prompt.user_prompt,
                    system=prompt.system_prompt,
                    temperature=0.3,
                    max_tokens=400,
                    timeout_s=self._response_llm_timeout_s(),
                    max_retries=self._response_llm_max_retries(),
                ):
                    if not chunk:
                        continue
                    reply_parts.append(chunk)
                    yield ChatStreamEvent(kind="reply_delta", stage="generation", text=chunk)
                reply = "".join(reply_parts).strip()
                if not reply:
                    llm_error = "empty_stream_response"
                    reply = self._build_response_fallback(
                        memory_context=memory_context,
                        user_message=user_message,
                        reason=llm_error,
                    )
                    yield ChatStreamEvent(kind="reply_delta", stage="generation", text=reply)
            except Exception as exc:
                llm_error = str(exc)
                reply = "".join(reply_parts).strip()
                if reply:
                    yield ChatStreamEvent(kind="status", stage="generation", text="流式生成中断，保留已生成内容")
                else:
                    reply = self._build_response_fallback(
                        memory_context=memory_context,
                        user_message=user_message,
                        reason=llm_error,
                    )
                    yield ChatStreamEvent(kind="status", stage="generation", text="生成失败，已切换到降级回复")
                    yield ChatStreamEvent(kind="reply_delta", stage="generation", text=reply)
        generation_ms = (time.perf_counter() - generation_started) * 1000.0

        yield ChatStreamEvent(kind="status", stage="ingest", text="正在写回记忆")
        ingest_started = time.perf_counter()
        # 此时将 LLM 的回复摄入引擎，作为其“说出过的话”
        ingest_result = self.ingest_interaction(
            event_id=resolved_event_id,
            session_id=session_id,
            user_message=user_message,
            agent_message=reply,
            actors=actors,
            context=context,
            ts=ts,
        )
        ingest_ms = (time.perf_counter() - ingest_started) * 1000.0

        total_ms = (time.perf_counter() - total_started) * 1000.0
        yield ChatStreamEvent(
            kind="done",
            stage="done",
            result=ChatTurnResult(
                reply=reply,
                event_id=resolved_event_id,
                memory_context=memory_context,
                rendered_memory_brief=prompt.rendered_memory_brief,
                system_prompt=prompt.system_prompt,
                user_prompt=prompt.user_prompt,
                retrieval_trace_summary=trace_summary,
                ingest_result=ingest_result,
                timings=ChatTimings(
                    retrieval_ms=retrieval_ms,
                    generation_ms=generation_ms,
                    ingest_ms=ingest_ms,
                    total_ms=total_ms,
                ),
                llm_error=llm_error,
            ),
        )

    def respond(
        self,
        *,
        session_id: str,
        user_message: str,
        event_id: Optional[str] = None,
        context: Optional[str] = None,
        actors: Optional[Sequence[str]] = None,
        k: int = 6,
        ts: Optional[float] = None,
    ) -> ChatTurnResult:
        """非流式对话流程。逻辑与 respond_stream 类似，但阻塞等待全部结果。"""
        total_started = time.perf_counter()
        resolved_event_id = event_id or f"evt_resp_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

        retrieval_started = time.perf_counter()
        memory_context, trace_summary = self.build_response_context(user_message=user_message, k=k)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0

        prompt = self.response_contexts.build_prompt(
            user_message=user_message,
            rendered_memory_brief=self.response_contexts.render_memory_brief(memory_context),
        )

        generation_started = time.perf_counter()
        llm_error: Optional[str] = None
        reply = self._generate_reply(prompt=prompt, memory_context=memory_context, user_message=user_message)
        if reply.startswith("__LLM_ERROR__:"):
            llm_error = reply.removeprefix("__LLM_ERROR__:")
            reply = self._build_response_fallback(
                memory_context=memory_context,
                user_message=user_message,
                reason=llm_error,
            )
        generation_ms = (time.perf_counter() - generation_started) * 1000.0

        ingest_started = time.perf_counter()
        ingest_result = self.ingest_interaction(
            event_id=resolved_event_id,
            session_id=session_id,
            user_message=user_message,
            agent_message=reply,
            actors=actors,
            context=context,
            ts=ts,
        )
        ingest_ms = (time.perf_counter() - ingest_started) * 1000.0

        total_ms = (time.perf_counter() - total_started) * 1000.0
        return ChatTurnResult(
            reply=reply,
            event_id=resolved_event_id,
            memory_context=memory_context,
            rendered_memory_brief=prompt.rendered_memory_brief,
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt,
            retrieval_trace_summary=trace_summary,
            ingest_result=ingest_result,
            timings=ChatTimings(
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                ingest_ms=ingest_ms,
                total_ms=total_ms,
            ),
            llm_error=llm_error,
        )

    def _generate_reply(self, *, prompt: Any, memory_context: StructuredMemoryContext, user_message: str) -> str:
        """调用 LLM 进行文本补全。"""
        if self.llm is None:
            return "__LLM_ERROR__:llm_not_configured"
        try:
            return self.llm.complete(
                prompt.user_prompt,
                system=prompt.system_prompt,
                temperature=0.3,
                max_tokens=400,
                timeout_s=self._response_llm_timeout_s(),
                max_retries=self._response_llm_max_retries(),
            ).strip()
        except Exception as exc:
            return f"__LLM_ERROR__:{exc}"

    @staticmethod
    def _build_response_fallback(
        *,
        memory_context: StructuredMemoryContext,
        user_message: str,
        reason: Optional[str] = None,
    ) -> str:
        """当 LLM 不可用时，利用灵魂引擎状态构建一个降级回复。"""
        if reason == "llm_not_configured":
            prefix = "当前还没有配置语言模型。"
        else:
            prefix = "这轮语言模型调用失败了。"
        if memory_context.narrative_summary is not None:
            return prefix + f"我现在还能稳住的内部状态是：{memory_context.narrative_summary.current_mode}。"
        return (
            f"{prefix} 当前没有足够稳定的 soul-memory 线索来回答。"
            f"我会先把这轮对话记住：{user_message[:80]}"
        )

    def query(self, *, text: str, k: int = 8) -> QueryResult:
        """显式记忆查询接口。"""
        with self._lock:
            trace = self.mem.query(text, k=k)
            hits: List[QueryHit] = []
            for node_id, score, kind in trace.ranked:
                snippet = self._snippet_for(node_id=node_id, kind=kind)
                metadata: Optional[Dict[str, str]] = None
                if kind == "plot":
                    plot = self.mem.plots.get(node_id)
                    if plot is not None:
                        metadata = {"source": plot.source, "story_id": plot.story_id or ""}
                hits.append(QueryHit(id=node_id, kind=kind, score=float(score), snippet=snippet, metadata=metadata))
        return QueryResult(query=text, attractor_path_len=len(trace.attractor_path), hits=hits)

    def _snippet_for(self, *, node_id: str, kind: str) -> str:
        """根据节点 ID 和类型获取可显示的文本片段。"""
        if kind == "plot":
            plot = self.mem.plots.get(node_id)
            return "" if plot is None else plot.text[:240]
        if kind == "story":
            story = self.mem.stories.get(node_id)
            if story is None:
                return ""
            return f"plots={len(story.plot_ids)} status={story.status} unresolved={story.unresolved_energy:.3f}"
        theme = self.mem.themes.get(node_id)
        if theme is None:
            return ""
        return (theme.name or theme.description)[:240]

    def feedback(self, *, query_text: str, chosen_id: str, success: bool) -> None:
        """用户检索反馈接口：微调度量学习和图置信度。"""
        with self._lock:
            self.mem.feedback_retrieval(query_text=query_text, chosen_id=chosen_id, success=success)

    def evolve(self, *, dreams: Optional[int] = None) -> List[Any]:
        """强制触发系统演化（做梦）。"""
        with self._lock:
            return self.mem.evolve(dreams=dreams or self.settings.dreams_per_evolve)

    def get_identity(self) -> Dict[str, Any]:
        """获取当前 Agent 的身份和叙事摘要。"""
        with self._lock:
            identity = self.mem.snapshot_identity()
            summary = self.mem.narrative_summary()
            return {
                "identity": identity.to_state_dict(),
                "narrative_summary": summary.to_state_dict(),
            }

    def get_stats(self) -> Dict[str, Any]:
        """获取运行时统计数据。"""
        with self._lock:
            return {
                "plot_count": len(self.mem.plots),
                "story_count": len(self.mem.stories),
                "theme_count": len(self.mem.themes),
                "current_mode": self.mem.identity.current_mode_label,
                "pressure": self.mem.identity.narrative_pressure(),
                "dream_count": self.mem.identity.dream_count,
                "repair_count": self.mem.identity.repair_count,
                "active_energy": self.mem.identity.active_energy,
                "repressed_energy": self.mem.identity.repressed_energy,
            }
