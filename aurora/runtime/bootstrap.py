from __future__ import annotations

import logging
import os
from typing import Optional

from aurora.core.soul_memory import (
    AuroraSoulMemory,
    HeuristicMeaningExtractor,
    LLMMeaningExtractor,
    SoulMemoryConfig,
)
from aurora.exceptions import ConfigurationError
from aurora.integrations.embeddings.base import EmbeddingProvider
from aurora.integrations.llm.mock import MockLLM
from aurora.integrations.llm.provider import LLMProvider
from aurora.runtime.settings import AuroraSettings

logger = logging.getLogger(__name__)


def check_embedding_api_keys() -> None:
    bailian_key = os.environ.get("AURORA_BAILIAN_EMBEDDING_API_KEY")
    ark_key = os.environ.get("AURORA_ARK_API_KEY")
    if bailian_key or ark_key:
        return
    logger.info(
        "No embedding API key found. Using LocalSemanticEmbedding. "
        "Set AURORA_EMBEDDING_PROVIDER to bailian, ark, local, or hash."
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


def create_embedding_provider(settings: AuroraSettings) -> EmbeddingProvider:
    provider = settings.embedding_provider
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


def build_memory_config(settings: AuroraSettings) -> SoulMemoryConfig:
    return SoulMemoryConfig(
        dim=settings.dim,
        metric_rank=settings.metric_rank,
        max_plots=settings.max_plots,
        kde_reservoir=settings.kde_reservoir,
        subconscious_reservoir=settings.subconscious_reservoir,
        story_alpha=settings.story_alpha,
        theme_alpha=settings.theme_alpha,
        gate_feature_dim=8,
        retrieval_kinds=("theme", "story", "plot"),
        phase_refractory_steps=4,
        initial_archetype=settings.initial_archetype,
    )


def create_meaning_extractor(*, settings: AuroraSettings, llm: Optional[LLMProvider]) -> HeuristicMeaningExtractor | LLMMeaningExtractor:
    if settings.meaning_extractor == "llm":
        if llm is None:
            raise ConfigurationError("LLM meaning extractor requires a live LLM provider")
        return LLMMeaningExtractor(
            llm,
            timeout_s=min(settings.llm_timeout, 12.0),
            max_retries=max(1, min(2, int(settings.llm_max_retries))),
        )
    return HeuristicMeaningExtractor()


def create_memory(*, settings: AuroraSettings, llm: Optional[LLMProvider] = None) -> AuroraSoulMemory:
    cfg = build_memory_config(settings)
    embedder = create_embedding_provider(settings)
    extractor = create_meaning_extractor(settings=settings, llm=llm)
    return AuroraSoulMemory(
        cfg=cfg,
        seed=int(settings.memory_seed),
        embedder=embedder,
        extractor=extractor,
    )
