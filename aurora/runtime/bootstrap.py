from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from aurora.core.memory import AuroraMemory
from aurora.core.models.config import MemoryConfig
from aurora.integrations.embeddings.base import EmbeddingProvider
from aurora.integrations.llm.mock import MockLLM
from aurora.integrations.llm.provider import LLMProvider
from aurora.runtime.settings import AuroraSettings

if TYPE_CHECKING:
    from aurora.integrations.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


def check_embedding_api_keys() -> None:
    bailian_key = os.environ.get("AURORA_BAILIAN_API_KEY")
    ark_key = os.environ.get("AURORA_ARK_API_KEY")

    if bailian_key or ark_key:
        if bailian_key:
            logger.info("AURORA_BAILIAN_API_KEY is configured")
        if ark_key:
            logger.info("AURORA_ARK_API_KEY is configured")
        return

    logger.info(
        "No embedding API key found. Using LocalSemanticEmbedding. "
        "Set AURORA_EMBEDDING_PROVIDER to bailian, ark, local, or hash."
    )


def create_llm_provider(settings: AuroraSettings) -> LLMProvider:
    if settings.llm_provider == "ark" and settings.ark_api_key:
        try:
            from aurora.integrations.llm.ark import ArkLLMWithFallback

            logger.info("Using Ark LLM provider with model: %s", settings.ark_llm_model)
            return ArkLLMWithFallback(
                api_key=settings.ark_api_key,
                model=settings.ark_llm_model,
                base_url=settings.ark_base_url,
                max_retries=settings.llm_max_retries,
                timeout=settings.llm_timeout,
            )
        except Exception as exc:
            logger.warning("Failed to create Ark LLM provider: %s; falling back to mock", exc)
    elif settings.llm_provider == "ark":
        logger.warning("Ark LLM provider selected but no API key provided; using mock")

    return MockLLM()


def create_embedding_provider(settings: AuroraSettings) -> EmbeddingProvider:
    provider = settings.embedding_provider

    if provider == "bailian" and settings.bailian_api_key:
        try:
            from aurora.integrations.embeddings.bailian import BailianEmbeddingWithFallback

            logger.info("Using Bailian embedding provider with model: %s", settings.bailian_embedding_model)
            return BailianEmbeddingWithFallback(
                api_key=settings.bailian_api_key,
                model=settings.bailian_embedding_model,
                base_url=settings.bailian_base_url,
                fallback_dim=settings.dim,
                use_cache=settings.embedding_cache_enabled,
                cache_size=settings.embedding_cache_size,
            )
        except Exception as exc:
            logger.warning("Failed to create Bailian embedding provider: %s; falling back to local", exc)

    if provider == "ark" and settings.ark_api_key:
        try:
            from aurora.integrations.embeddings.ark import ArkEmbeddingWithFallback

            logger.info("Using Ark embedding provider")
            return ArkEmbeddingWithFallback(
                api_key=settings.ark_api_key,
                fallback_dim=settings.dim,
                use_cache=settings.embedding_cache_enabled,
                cache_size=settings.embedding_cache_size,
            )
        except Exception as exc:
            logger.warning("Failed to create Ark embedding provider: %s; falling back to local", exc)

    if provider == "hash":
        from aurora.integrations.embeddings.hash import HashEmbedding

        logger.info("Using HashEmbedding for deterministic testing")
        return HashEmbedding(dim=settings.dim)

    from aurora.integrations.embeddings.local_semantic import LocalSemanticEmbedding

    logger.info("Using LocalSemanticEmbedding")
    return LocalSemanticEmbedding(dim=settings.dim)


def build_memory_config(settings: AuroraSettings) -> MemoryConfig:
    return MemoryConfig(
        dim=settings.dim,
        metric_rank=settings.metric_rank,
        max_plots=settings.max_plots,
        kde_reservoir=settings.kde_reservoir,
        story_alpha=settings.story_alpha,
        theme_alpha=settings.theme_alpha,
    )


def create_memory(*, settings: AuroraSettings, user_id: str) -> AuroraMemory:
    seed = abs(hash(user_id)) % (2**32)
    cfg = build_memory_config(settings)
    embedder = create_embedding_provider(settings)
    return AuroraMemory(cfg=cfg, seed=int(seed), embedder=embedder, benchmark_mode=cfg.benchmark_mode)
