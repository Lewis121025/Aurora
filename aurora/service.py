"""
AURORA 遗留服务层
================

注意：这是原始的单体服务模块。
对于新项目，请考虑使用 aurora.services/ 中的 CQRS 架构：

    from aurora.services import IngestionService, QueryService, IngestWorker

该模块为了向后兼容而维护，仍被以下模块使用：
- aurora.hub（多租户路由）
- aurora.mcp.server（MCP 协议）
- 各种测试

新的 services/ 架构提供：
- 更好的水平扩展（分离的读写路径）
- 非阻塞写入（基于队列的摄入）
- 更低的读取延迟（优化的查询路径）
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from aurora.config import AuroraSettings
from aurora.privacy.pii import redact
from aurora.storage.event_log import Event, SQLiteEventLog
from aurora.storage.snapshot import Snapshot, SnapshotStore
from aurora.storage.doc_store import Document, SQLiteDocStore
from aurora.utils.logging import log_event

from aurora.llm.provider import LLMProvider
from aurora.llm.mock import MockLLM
from aurora.llm import prompts
from aurora.llm.schemas import PlotExtraction

from aurora.algorithms.aurora_core import AuroraMemory
from aurora.algorithms.models.config import MemoryConfig
from aurora.algorithms.models.trace import QueryHit
from aurora.algorithms.components.metric import LowRankMetric
from aurora.algorithms.causal import CausalMemoryGraph, CausalEdgeBelief
from aurora.algorithms.coherence import CoherenceGuardian, CoherenceReport
from aurora.algorithms.self_narrative import SelfNarrativeEngine, SelfNarrative

logger = logging.getLogger(__name__)


def check_embedding_api_keys() -> None:
    """检查嵌入 API 密钥是否已配置，如果未配置则打印指导。

    检查项：
    - AURORA_BAILIAN_API_KEY: 阿里云百炼 embedding API
    - AURORA_ARK_API_KEY: 火山方舟 embedding API

    如果都未设置，打印清晰的配置指南。
    """
    bailian_key = os.environ.get("AURORA_BAILIAN_API_KEY")
    ark_key = os.environ.get("AURORA_ARK_API_KEY")
    
    if bailian_key or ark_key:
        # 至少配置了一个 API 密钥
        if bailian_key:
            logger.info("✓ AURORA_BAILIAN_API_KEY is configured")
        if ark_key:
            logger.info("✓ AURORA_ARK_API_KEY is configured")
        return

    # 两个 API 密钥都未配置 - 打印信息消息
    info_msg = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ℹ️  USING LOCAL SEMANTIC EMBEDDING ℹ️                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  No embedding API key found. Using LocalSemanticEmbedding for basic testing. ║
║  LocalSemanticEmbedding captures word-level semantics (better than random).  ║
║                                                                              ║
║  For production, configure one of the following embedding providers:         ║
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │ Option 1: 阿里云百炼 (Alibaba Bailian) - Recommended                    │ ║
║  │                                                                         │ ║
║  │   export AURORA_BAILIAN_API_KEY="your-bailian-api-key"                  │ ║
║  │   export AURORA_EMBEDDING_PROVIDER="bailian"                            │ ║
║  │                                                                         │ ║
║  │   Get API key: https://bailian.console.aliyun.com/                      │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │ Option 2: 火山方舟 (Volcengine Ark)                                     │ ║
║  │                                                                         │ ║
║  │   export AURORA_ARK_API_KEY="your-ark-api-key"                          │ ║
║  │   export AURORA_EMBEDDING_PROVIDER="ark"                                │ ║
║  │                                                                         │ ║
║  │   Get API key: https://console.volcengine.com/ark                       │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                              ║
║  You can also set these in a .env file at the project root.                  ║
║  See .env.example for a template.                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    logger.info(info_msg)


def create_llm_provider(settings: AuroraSettings) -> LLMProvider:
    """根据设置创建 LLM 提供者。

    支持：
    - "ark": 火山方舟 (Volcengine Ark) - 需要 ark_api_key
    - "mock": 本地模拟用于测试
    """
    if settings.llm_provider == "ark" and settings.ark_api_key:
        try:
            from aurora.llm.ark import ArkLLMWithFallback
            logger.info(f"Using Ark LLM provider with model: {settings.ark_llm_model}")
            return ArkLLMWithFallback(
                api_key=settings.ark_api_key,
                model=settings.ark_llm_model,
                base_url=settings.ark_base_url,
                max_retries=settings.llm_max_retries,
                timeout=settings.llm_timeout,
            )
        except Exception as e:
            logger.warning(f"Failed to create Ark LLM provider: {e}, falling back to mock")
            return MockLLM()
    else:
        if settings.llm_provider == "ark" and not settings.ark_api_key:
            logger.warning("Ark LLM provider selected but no API key provided, using mock")
        return MockLLM()


def create_embedding_provider(settings: AuroraSettings):
    """根据设置创建嵌入提供者。

    支持：
    - "bailian": 阿里云百炼 (Alibaba Bailian) - 需要 bailian_api_key
    - "ark": 火山方舟 (Volcengine Ark) - 需要 ark_api_key + endpoint
    - "local": 本地语义嵌入（词向量，捕获基本语义）
    - "hash": 本地哈希嵌入用于遗留测试（随机向量，无语义）

    当未配置 API 密钥时，默认为 "local"（LocalSemanticEmbedding）
    提供基本语义相似性，不同于 HashEmbedding。
    """
    # 阿里云百炼
    if settings.embedding_provider == "bailian" and settings.bailian_api_key:
        try:
            from aurora.embeddings.bailian import BailianEmbeddingWithFallback
            logger.info(f"Using Bailian embedding provider with model: {settings.bailian_embedding_model}")
            return BailianEmbeddingWithFallback(
                api_key=settings.bailian_api_key,
                model=settings.bailian_embedding_model,
                base_url=settings.bailian_base_url,
                fallback_dim=settings.dim,
                use_cache=settings.embedding_cache_enabled,
                cache_size=settings.embedding_cache_size,
            )
        except Exception as e:
            logger.warning(f"Failed to create Bailian embedding provider: {e}, falling back to local semantic")

    # 火山方舟（需要 endpoint ID）
    elif settings.embedding_provider == "ark" and settings.ark_api_key:
        try:
            from aurora.embeddings.ark import ArkEmbeddingWithFallback
            logger.info(f"Using Ark embedding provider")
            return ArkEmbeddingWithFallback(
                api_key=settings.ark_api_key,
                fallback_dim=settings.dim,
                use_cache=settings.embedding_cache_enabled,
                cache_size=settings.embedding_cache_size,
            )
        except Exception as e:
            logger.warning(f"Failed to create Ark embedding provider: {e}, falling back to local semantic")
    
    # 显式指定 hash embedding（向后兼容）
    elif settings.embedding_provider == "hash":
        from aurora.embeddings.hash import HashEmbedding
        logger.info("Using local Hash embedding provider (random vectors, no semantic meaning)")
        return HashEmbedding(dim=settings.dim)

    # 本地语义嵌入（默认，或显式指定 "local"）
    # 比 HashEmbedding 更好：语义相似的文本会有相似的向量
    from aurora.embeddings.local_semantic import LocalSemanticEmbedding
    logger.info("Using local Semantic embedding provider (word vectors, captures basic semantics)")
    return LocalSemanticEmbedding(dim=settings.dim)


@dataclass
class IngestResult:
    event_id: str
    plot_id: str
    story_id: Optional[str]
    encoded: bool
    # 轻量级信号
    tension: float
    surprise: float
    pred_error: float
    redundancy: float


@dataclass
class QueryResult:
    query: str
    attractor_path_len: int
    hits: List[QueryHit]


@dataclass
class CoherenceResult:
    overall_score: float
    conflict_count: int
    unfinished_story_count: int
    recommendations: List[str]


class AuroraTenant:
    """每个用户的内存实例。

    生产环保：
    - 并发性：由锁保护。
    - 持久性：事件溯源 + 定期快照。
    - 幂等性：event_id 唯一；重复事件不会重新摄入。

    扩展功能：
    - 因果推断
    - 一致性检查
    - 自我叙述管理
    """

    def __init__(self, *, user_id: str, settings: AuroraSettings, llm: Optional[LLMProvider] = None):
        self.user_id = user_id
        self.settings = settings
        self.llm: LLMProvider = llm or create_llm_provider(settings)
        
        # Check for embedding API configuration at startup
        check_embedding_api_keys()

        self._lock = threading.RLock()

        # 租户目录
        self.user_dir = os.path.join(self.settings.data_dir, f"user_{self.user_id}")
        os.makedirs(self.user_dir, exist_ok=True)

        # 存储
        self.event_log = SQLiteEventLog(os.path.join(self.user_dir, self.settings.event_log_filename))
        self.doc_store = SQLiteDocStore(os.path.join(self.user_dir, "docs.sqlite3"))
        self.snapshots = SnapshotStore(os.path.join(self.user_dir, self.settings.snapshot_dirname))

        # 状态
        self.last_seq: int = 0
        self.mem: AuroraMemory = self._load_or_init()

        # 扩展模块
        self.coherence_guardian = CoherenceGuardian(self.mem.metric)
        self.self_narrative_engine = SelfNarrativeEngine(self.mem.metric)

        # 因果信念（单独存储以提高灵活性）
        self.causal_beliefs: Dict[tuple, CausalEdgeBelief] = {}

        # 重放快照后的任何事件
        self._replay()

    # ---------------------------------------------------------------------
    # 引导：快照 + 重放
    # ---------------------------------------------------------------------

    def _load_or_init(self) -> AuroraMemory:
        latest = self.snapshots.latest()
        if latest is not None:
            _seq, snap = latest
            self.last_seq = snap.last_seq
            return snap.state

        # 每个用户的确定性种子用于重放稳定性
        seed = abs(hash(self.user_id)) % (2**32)
        cfg = MemoryConfig(
            dim=self.settings.dim,
            metric_rank=self.settings.metric_rank,
            max_plots=self.settings.max_plots,
            kde_reservoir=self.settings.kde_reservoir,
            story_alpha=self.settings.story_alpha,
            theme_alpha=self.settings.theme_alpha,
        )
        return AuroraMemory(cfg=cfg, seed=int(seed))

    def _replay(self) -> None:
        # 重放 last_seq 之后的事件
        for seq, ev in self.event_log.iter_events(after_seq=self.last_seq, user_id=self.user_id):
            if ev.type != "interaction":
                continue
            payload = ev.payload
            self._apply_interaction(
                event_id=ev.id,
                user_message=payload.get("user_message", ""),
                agent_message=payload.get("agent_message", ""),
                actors=payload.get("actors"),
                context=payload.get("context"),
                ts=ev.ts,
                persist=False,  # already persisted
            )
            self.last_seq = seq

    # ---------------------------------------------------------------------
    # 公共 API
    # ---------------------------------------------------------------------

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
        logger: Optional[Any] = None,
    ) -> IngestResult:
        ts = ts or time.time()

        with self._lock:
            # 幂等性：如果事件存在，返回存储的摄入结果
            existing_seq = self.event_log.get_seq_by_id(event_id)
            if existing_seq is not None:
                doc = self.doc_store.get(f"ingest:{event_id}")
                if doc:
                    body = doc.body
                    return IngestResult(
                        event_id=event_id,
                        plot_id=body["plot_id"],
                        story_id=body.get("story_id"),
                        encoded=bool(body.get("encoded", True)),
                        tension=float(body.get("tension", 0.0)),
                        surprise=float(body.get("surprise", 0.0)),
                        pred_error=float(body.get("pred_error", 0.0)),
                        redundancy=float(body.get("redundancy", 0.0)),
                    )
                # 回退：视为无操作
                return IngestResult(
                    event_id=event_id,
                    plot_id="",
                    story_id=None,
                    encoded=False,
                    tension=0.0,
                    surprise=0.0,
                    pred_error=0.0,
                    redundancy=0.0,
                )

            # 隐私钩子
            if self.settings.pii_redaction_enabled:
                user_message = redact(user_message).redacted_text
                agent_message = redact(agent_message).redacted_text

            # 先持久化事件（预写日志）
            seq = self.event_log.append(
                Event(
                    id=event_id,
                    ts=ts,
                    user_id=self.user_id,
                    session_id=session_id,
                    type="interaction",
                    payload={
                        "user_message": user_message,
                        "agent_message": agent_message,
                        "actors": list(actors) if actors else None,
                        "context": context,
                    },
                )
            )

            # 应用到内存
            res = self._apply_interaction(
                event_id=event_id,
                user_message=user_message,
                agent_message=agent_message,
                actors=actors,
                context=context,
                ts=ts,
                persist=True,
            )
            self.last_seq = max(self.last_seq, seq)

            # 快照策略（操作性）
            if self.settings.snapshot_every_events > 0 and self.last_seq % self.settings.snapshot_every_events == 0:
                self._snapshot(logger=logger)

            if logger:
                log_event(logger, "aurora_ingest", user_id=self.user_id, event_id=event_id, plot_id=res.plot_id)

            return res

    def query(self, *, text: str, k: int = 8) -> QueryResult:
        with self._lock:
            trace = self.mem.query(text, k=k)

        hits: List[QueryHit] = []
        for nid, score, kind in trace.ranked:
            snippet = ""
            if kind == "plot":
                p = self.mem.plots.get(nid)
                snippet = (p.text[:240] + "...") if p else ""
            else:
                # 尝试文档存储摘要
                doc = self.doc_store.get(nid)
                if doc:
                    snippet = (doc.body.get("summary", "") or doc.body.get("description", ""))[:240]
            hits.append(QueryHit(id=nid, kind=kind, score=float(score), snippet=snippet))

        return QueryResult(query=text, attractor_path_len=len(trace.attractor_path), hits=hits)

    def feedback(self, *, query_text: str, chosen_id: str, success: bool) -> None:
        with self._lock:
            self.mem.feedback_retrieval(query_text=query_text, chosen_id=chosen_id, success=success)

    def evolve(self, *, logger: Optional[Any] = None) -> None:
        with self._lock:
            self.mem.evolve()

            # 从主题更新自我叙述
            themes = list(self.mem.themes.values())
            self.self_narrative_engine.update_from_themes(themes)
            
        if logger:
            log_event(logger, "aurora_evolve", user_id=self.user_id, stories=len(self.mem.stories), themes=len(self.mem.themes))

    def check_coherence(self, *, logger: Optional[Any] = None) -> CoherenceResult:
        """检查内存一致性并返回报告"""
        with self._lock:
            report = self.coherence_guardian.full_check(
                graph=self.mem.graph,
                plots=self.mem.plots,
                stories=self.mem.stories,
                themes=self.mem.themes,
                causal_beliefs=self.causal_beliefs,
            )
        
        recommendations = [r.action_description for r in report.recommended_actions[:5]]
        
        result = CoherenceResult(
            overall_score=report.overall_score,
            conflict_count=len(report.conflicts),
            unfinished_story_count=len(report.unfinished_stories),
            recommendations=recommendations,
        )
        
        if logger:
            log_event(
                logger, "aurora_coherence_check",
                user_id=self.user_id,
                score=result.overall_score,
                conflicts=result.conflict_count,
            )
        
        return result

    def get_self_narrative(self) -> Dict[str, Any]:
        """获取当前自我叙述"""
        with self._lock:
            narrative = self.self_narrative_engine.narrative
            
            return {
                "identity_statement": narrative.identity_statement,
                "identity_narrative": narrative.identity_narrative,
                "capability_narrative": narrative.capability_narrative,
                "core_values": narrative.core_values,
                "coherence_score": narrative.coherence_score,
                "capabilities": {
                    name: {
                        "probability": cap.capability_probability(),
                        "description": cap.description,
                    }
                    for name, cap in narrative.capabilities.items()
                },
                "relationships": {
                    entity_id: {
                        "trust": rel.trust(),
                        "familiarity": rel.familiarity(),
                        "interaction_count": rel.interaction_count,
                    }
                    for entity_id, rel in narrative.relationships.items()
                },
                "unresolved_tensions": narrative.unresolved_tensions,
                "full_narrative": narrative.to_full_narrative(),
            }

    def get_causal_chain(self, node_id: str, direction: str = "ancestors") -> List[Dict[str, Any]]:
        """获取节点的因果祖先或后代"""
        with self._lock:
            # 构建因果内存图视图
            causal_graph = CausalMemoryGraph(self.mem.metric)
            causal_graph.g = self.mem.graph.g.copy()
            causal_graph.causal_beliefs = self.causal_beliefs
            
            if direction == "ancestors":
                chain = causal_graph.get_causal_ancestors(node_id)
            else:
                chain = causal_graph.get_causal_descendants(node_id)
            
            return [
                {"node_id": nid, "strength": strength}
                for nid, strength in chain
            ]

    def record_feedback_with_learning(
        self,
        *,
        query_text: str,
        chosen_id: str,
        success: bool,
        entity_id: Optional[str] = None,
    ) -> None:
        """记录反馈并更新学习模型"""
        with self._lock:
            # 标准反馈
            self.mem.feedback_retrieval(
                query_text=query_text,
                chosen_id=chosen_id,
                success=success,
            )

            # 从交互更新自我叙述
            plot = self.mem.plots.get(chosen_id)
            if plot:
                self.self_narrative_engine.update_from_interaction(
                    plot=plot,
                    success=success,
                    entity_id=entity_id or self.user_id,
                )

            # 如果适用，更新因果信念
            # （可以追踪检索中使用的边并更新它们）

    # ---------------------------------------------------------------------
    # 内部辅助函数
    # ---------------------------------------------------------------------

    def _apply_interaction(
        self,
        *,
        event_id: str,
        user_message: str,
        agent_message: str,
        actors: Optional[Sequence[str]],
        context: Optional[str],
        ts: float,
        persist: bool,
    ) -> IngestResult:
        # 1) 可选的结构化提取（保持便宜；在生产中可以异步）
        extraction = self._extract_plot(user_message=user_message, agent_message=agent_message, context=context)

        # 2) 为算法核心构建交互文本
        interaction_text = f"USER: {user_message}\nAGENT: {agent_message}\nOUTCOME: {extraction.outcome}".strip()

        plot = self.mem.ingest(
            interaction_text,
            actors=actors or extraction.actors,
            context_text=context,
            event_id=event_id,
        )

        encoded = plot.id in self.mem.plots

        # 3) 持久化派生文档
        if persist:
            self.doc_store.upsert(
                Document(
                    id=plot.id,
                    kind="plot",
                    user_id=self.user_id,
                    ts=ts,
                    body={
                        "schema_version": extraction.schema_version,
                        "actors": extraction.actors,
                        "action": extraction.action,
                        "context": extraction.context,
                        "outcome": extraction.outcome,
                        "goal": extraction.goal,
                        "obstacles": extraction.obstacles,
                        "decision": extraction.decision,
                        "emotion_valence": extraction.emotion_valence,
                        "emotion_arousal": extraction.emotion_arousal,
                        "claims": [c.model_dump() for c in extraction.claims],
                        "raw": {"user_message": user_message, "agent_message": agent_message},
                    },
                )
            )

            self.doc_store.upsert(
                Document(
                    id=f"ingest:{event_id}",
                    kind="ingest_result",
                    user_id=self.user_id,
                    ts=ts,
                    body={
                        "event_id": event_id,
                        "plot_id": plot.id,
                        "story_id": plot.story_id,
                        "encoded": encoded,
                        "tension": plot.tension,
                        "surprise": plot.surprise,
                        "pred_error": plot.pred_error,
                        "redundancy": plot.redundancy,
                    },
                )
            )

        return IngestResult(
            event_id=event_id,
            plot_id=plot.id,
            story_id=plot.story_id,
            encoded=encoded,
            tension=float(plot.tension),
            surprise=float(plot.surprise),
            pred_error=float(plot.pred_error),
            redundancy=float(plot.redundancy),
        )

    def _extract_plot(self, *, user_message: str, agent_message: str, context: Optional[str]) -> PlotExtraction:
        # 在生产中：应用路由和缓存；也允许"无 LLM"模式。
        instruction = prompts.instruction("PlotExtraction")
        u = prompts.render(
            prompts.PLOT_EXTRACTION_USER,
            instruction=instruction,
            user_message=user_message,
            agent_message=agent_message,
            context=context or "",
        )
        try:
            return self.llm.complete_json(
                system=prompts.PLOT_EXTRACTION_SYSTEM,
                user=u,
                schema=PlotExtraction,
                temperature=0.2,
                timeout_s=20.0,
            )
        except Exception as e:
            logger.debug(f"LLM plot extraction failed, using minimal fallback: {e}")
            return PlotExtraction(action=user_message[:120], actors=["user", "agent"])

    def _snapshot(self, *, logger: Optional[Any] = None) -> None:
        snap = Snapshot(last_seq=self.last_seq, state=self.mem)
        path = self.snapshots.save(snap)
        if logger:
            log_event(logger, "aurora_snapshot", user_id=self.user_id, last_seq=self.last_seq, path=path)
