from __future__ import annotations

import logging
import os
from typing import Optional

from aurora.integrations.embeddings.base import EmbeddingProvider
from aurora.integrations.llm.mock import MockLLM
from aurora.integrations.llm.provider import LLMProvider
from aurora.runtime.settings import AuroraSettings
from aurora.soul.engine import AuroraSoul, SoulConfig
from aurora.soul.extractors import (
    CombinatorialNarrativeProvider,
    HeuristicMeaningProvider,
    LLMMeaningProvider,
    LLMNarrativeProvider,
)
from aurora.system.errors import ConfigurationError

logger = logging.getLogger(__name__)


def check_embedding_api_keys() -> None:
    bailian_key = os.environ.get("AURORA_BAILIAN_EMBEDDING_API_KEY")
    ark_key = os.environ.get("AURORA_ARK_API_KEY")
    if bailian_key or ark_key:
        return
    logger.info(
        "No remote embedding API key found. Aurora v4 will use LocalSemanticEmbedding unless you set AURORA_EMBEDDING_PROVIDER."
    )


def create_llm_provider(settings: AuroraSettings) -> LLMProvider:
    if settings.llm_provider == "bailian":
        if not settings.bailian_llm_api_key:
            raise ConfigurationError("Bailian LLM provider selected but AURORA_BAILIAN_LLM_API_KEY is not set")
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

    return MockLLM()


def create_embedding_provider(settings: AuroraSettings, *, provider_override: Optional[str] = None) -> EmbeddingProvider:
    provider = provider_override or settings.embedding_provider
    if provider == "bailian":
        if not settings.bailian_embedding_api_key:
            raise ConfigurationError("Bailian embedding provider selected but AURORA_BAILIAN_EMBEDDING_API_KEY is not set")
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
            raise ConfigurationError("Ark embedding provider selected but AURORA_ARK_API_KEY is not set")
        from aurora.integrations.embeddings.ark import ArkEmbedding

        return ArkEmbedding(
            api_key=settings.ark_api_key,
            dimension=settings.dim,
            use_cache=settings.embedding_cache_enabled,
            cache_size=settings.embedding_cache_size,
        )

    if provider == "hash":
        from aurora.integrations.embeddings.hash import HashEmbedding

        return HashEmbedding(dim=settings.dim)

    from aurora.integrations.embeddings.local_semantic import LocalSemanticEmbedding

    return LocalSemanticEmbedding(dim=settings.dim)


def build_memory_config(settings: AuroraSettings) -> SoulConfig:
    return SoulConfig(
        dim=settings.dim,
        metric_rank=settings.metric_rank,
        max_plots=settings.max_plots,
        kde_reservoir=settings.kde_reservoir,
        subconscious_reservoir=settings.subconscious_reservoir,
        story_alpha=settings.story_alpha,
        theme_alpha=settings.theme_alpha,
        gate_feature_dim=8,
        retrieval_kinds=("theme", "story", "plot"),
        mode_refractory_steps=settings.mode_refractory_steps,
        mode_new_threshold=settings.mode_new_threshold,
        encode_min_events_before_gating=settings.encode_min_events_before_gating,
        max_recent_texts=settings.max_recent_texts,
        profile_text=settings.profile_text,
        persona_axes_json=settings.persona_axes_json,
        axis_merge_every_events=settings.axis_merge_every_events,
        persona_axis_budget=settings.persona_axis_budget,
    )


def create_meaning_provider(*, settings: AuroraSettings, llm: Optional[LLMProvider]):
    if settings.meaning_provider == "llm":
        if llm is None:
            raise ConfigurationError("LLM meaning provider requires a live LLM provider")
        return LLMMeaningProvider(
            llm,
            fallback=HeuristicMeaningProvider(),
            timeout_s=min(settings.llm_timeout, 12.0),
            max_retries=max(1, min(2, int(settings.llm_max_retries))),
        )
    return HeuristicMeaningProvider()


def create_narrative_provider(*, settings: AuroraSettings, llm: Optional[LLMProvider]):
    if settings.narrative_provider == "llm":
        if llm is None:
            raise ConfigurationError("LLM narrative provider requires a live LLM provider")
        return LLMNarrativeProvider(
            llm,
            fallback=CombinatorialNarrativeProvider(),
            timeout_s=min(settings.llm_timeout, 12.0),
            max_retries=max(1, min(2, int(settings.llm_max_retries))),
        )
    return CombinatorialNarrativeProvider()


def create_memory(*, settings: AuroraSettings, llm: Optional[LLMProvider] = None) -> AuroraSoul:
    cfg = build_memory_config(settings)
    event_embedder = create_embedding_provider(settings)
    axis_embedder = create_embedding_provider(
        settings,
        provider_override=settings.axis_embedding_provider or settings.embedding_provider,
    )
    meaning_provider = create_meaning_provider(settings=settings, llm=llm)
    narrator = create_narrative_provider(settings=settings, llm=llm)
    return AuroraSoul(
        cfg=cfg,
        seed=int(settings.memory_seed),
        event_embedder=event_embedder,
        axis_embedder=axis_embedder,
        meaning_provider=meaning_provider,
        narrator=narrator,
    )
