"""
aurora/runtime/bootstrap.py
启动引导模块：负责系统组件的初始化、配置加载以及依赖注入。
它将 settings 中的配置项转换为具体的 Provider 实例（LLM, Embedding, Memory）。
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from aurora.integrations.embeddings.base import EmbeddingProvider
from aurora.integrations.llm.provider import LLMProvider
from aurora.runtime.settings import AuroraSettings
from aurora.soul.engine import AuroraSoul, SoulConfig
from aurora.soul.query import QueryAnalyzer
from aurora.soul.extractors import (
    CombinatorialNarrativeProvider,
    HeuristicMeaningProvider,
    LLMMeaningProvider,
    LLMNarrativeProvider,
    MeaningProvider,
    NarrativeProvider,
)
from aurora.system.errors import ConfigurationError

logger = logging.getLogger(__name__)


def check_embedding_api_keys() -> None:
    """检查环境变量中是否存在远程 Embedding 服务的 API Key，若无则提示将使用本地模型。"""
    bailian_key = os.environ.get("AURORA_BAILIAN_EMBEDDING_API_KEY")
    ark_key = os.environ.get("AURORA_ARK_API_KEY")
    if bailian_key or ark_key:
        return
    logger.info(
        "No remote embedding API key found. Aurora v5 will use LocalSemanticEmbedding unless you set AURORA_EMBEDDING_PROVIDER."
    )


def create_llm_provider(settings: AuroraSettings) -> Optional[LLMProvider]:
    """根据 settings 创建并返回 LLM 提供者实例。"""
    if settings.llm_provider == "bailian":
        if not settings.bailian_llm_api_key:
            raise ConfigurationError(
                "Bailian LLM provider selected but AURORA_BAILIAN_LLM_API_KEY is not set"
            )
        from aurora.integrations.llm.bailian import BailianLLM

        return BailianLLM(
            api_key=settings.bailian_llm_api_key,
            model=settings.bailian_llm_model,
            base_url=settings.bailian_llm_base_url,
            max_retries=settings.llm_max_retries,
            timeout=settings.llm_timeout,
        )

    if settings.llm_provider == "ark":
        if not settings.ark_api_key:
            raise ConfigurationError("Ark LLM provider selected but AURORA_ARK_API_KEY is not set")
        from aurora.integrations.llm.ark import ArkLLM

        return ArkLLM(
            api_key=settings.ark_api_key,
            model=settings.ark_llm_model,
            base_url=settings.ark_base_url,
            max_retries=settings.llm_max_retries,
            timeout=settings.llm_timeout,
        )

    return None


def create_embedding_provider(
    settings: AuroraSettings, *, provider_override: Optional[str] = None
) -> EmbeddingProvider:
    """根据 settings 创建并返回 Embedding 提供者实例。支持缓存配置。"""
    provider = provider_override or settings.embedding_provider
    if provider == "bailian":
        if not settings.bailian_embedding_api_key:
            raise ConfigurationError(
                "Bailian embedding provider selected but AURORA_BAILIAN_EMBEDDING_API_KEY is not set"
            )
        from aurora.integrations.embeddings.bailian import BailianEmbedding

        return BailianEmbedding(
            api_key=settings.bailian_embedding_api_key,
            model=settings.bailian_embedding_model,
            base_url=settings.bailian_embedding_base_url,
            dimension=settings.dim,
            use_cache=settings.embedding_cache_enabled,
            cache_size=settings.embedding_cache_size,
        )

    if provider == "ark":
        if not settings.ark_api_key:
            raise ConfigurationError(
                "Ark embedding provider selected but AURORA_ARK_API_KEY is not set"
            )
        from aurora.integrations.embeddings.ark import ArkEmbedding

        return ArkEmbedding(
            api_key=settings.ark_api_key,
            dimension=settings.dim,
            use_cache=settings.embedding_cache_enabled,
            cache_size=settings.embedding_cache_size,
        )

    if provider == "hash":
        # 基于分词和哈希的简易 Embedding，无需外部 API。
        from aurora.integrations.embeddings.hash import HashEmbedding

        return HashEmbedding(dim=settings.dim)

    from aurora.integrations.embeddings.local_semantic import LocalSemanticEmbedding

    # 默认使用本地语义 Embedding。
    return LocalSemanticEmbedding(dim=settings.dim)


def build_memory_config(settings: AuroraSettings) -> SoulConfig:
    """将 AuroraSettings 转换为 soul 引擎专用的 SoulConfig 对象。"""
    return SoulConfig(
        dim=settings.dim,
        metric_rank=settings.metric_rank,
        max_plots=settings.max_plots,
        kde_reservoir=settings.kde_reservoir,
        retrieval_kinds=("summary", "theme", "story", "plot"),
        max_recent_texts=settings.max_recent_texts,
        profile_text=settings.profile_text,
        persona_axes_json=settings.persona_axes_json,
        axis_merge_every_events=settings.axis_merge_every_events,
        persona_axis_budget=settings.persona_axis_budget,
        graph_temporal_neighbors=settings.graph_temporal_neighbors,
        graph_semantic_neighbors=settings.graph_semantic_neighbors,
        graph_contradiction_neighbors=settings.graph_contradiction_neighbors,
        graph_similarity_threshold=settings.graph_similarity_threshold,
        graph_contradiction_threshold=settings.graph_contradiction_threshold,
        community_refresh_every_plots=settings.community_refresh_every_plots,
        dream_walk_steps=settings.dream_walk_steps,
        dream_walk_samples=settings.dream_walk_samples,
        dream_persist_threshold=settings.dream_persist_threshold,
    )


def create_meaning_provider(
    *, settings: AuroraSettings, llm: Optional[LLMProvider]
) -> MeaningProvider:
    """创建意义提取器。若配置为 'llm' 则使用大模型，否则使用启发式规则。"""
    if settings.meaning_provider == "llm":
        if llm is None:
            raise ConfigurationError("LLM meaning provider requires a live LLM provider")
        return LLMMeaningProvider(
            llm,
            fallback=HeuristicMeaningProvider(),
            timeout_s=min(settings.llm_timeout, 10.0),
            max_retries=1,
        )
    return HeuristicMeaningProvider()


def create_narrative_provider(
    *, settings: AuroraSettings, llm: Optional[LLMProvider]
) -> NarrativeProvider:
    """创建叙事生成器。负责生成自我总结、梦境和修复文本。"""
    if settings.narrative_provider == "llm":
        if llm is None:
            raise ConfigurationError("LLM narrative provider requires a live LLM provider")
        return LLMNarrativeProvider(
            llm,
            fallback=CombinatorialNarrativeProvider(),
            timeout_s=min(settings.llm_timeout, 10.0),
            max_retries=1,
        )
    return CombinatorialNarrativeProvider()


def create_query_analyzer(*, settings: AuroraSettings, llm: Optional[LLMProvider]) -> QueryAnalyzer:
    """创建唯一主链路的 LLM query router。"""
    if llm is None:
        raise ConfigurationError("LLM query analyzer requires a live LLM provider")
    return QueryAnalyzer(
        llm=llm,
        timeout_s=min(settings.llm_timeout, 5.0),
        max_retries=1,
    )


def create_memory(*, settings: AuroraSettings, llm: Optional[LLMProvider] = None) -> AuroraSoul:
    """
    全量引导函数：根据配置初始化 AuroraSoul 引擎的所有依赖。
    包含：
    1. 内存配置构建。
    2. 事件与轴 Embedding 提供者创建。
    3. 语义提取器与叙事提供者注入。
    """
    cfg = build_memory_config(settings)
    event_embedder = create_embedding_provider(settings)
    # 轴 Embedding 可以与普通事件 Embedding 使用不同的模型/服务。
    axis_embedder = create_embedding_provider(
        settings,
        provider_override=settings.axis_embedding_provider or settings.embedding_provider,
    )
    meaning_provider = create_meaning_provider(settings=settings, llm=llm)
    narrator = create_narrative_provider(settings=settings, llm=llm)
    query_analyzer = create_query_analyzer(settings=settings, llm=llm)

    return AuroraSoul(
        cfg=cfg,
        seed=int(settings.memory_seed),
        event_embedder=event_embedder,
        axis_embedder=axis_embedder,
        meaning_provider=meaning_provider,
        narrator=narrator,
        query_analyzer=query_analyzer,
    )
