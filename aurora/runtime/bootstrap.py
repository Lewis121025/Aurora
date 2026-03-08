from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from aurora.core.memory import AuroraMemory
from aurora.core.models.config import MemoryConfig
from aurora.exceptions import ConfigurationError
from aurora.integrations.embeddings.base import EmbeddingProvider
from aurora.integrations.llm.mock import MockLLM
from aurora.integrations.llm.provider import LLMProvider
from aurora.runtime.settings import AuroraSettings

if TYPE_CHECKING:
    from aurora.integrations.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


def check_embedding_api_keys() -> None:
    bailian_key = os.environ.get("AURORA_BAILIAN_EMBEDDING_API_KEY")
    ark_key = os.environ.get("AURORA_ARK_API_KEY")

    if bailian_key or ark_key:
        if bailian_key:
            logger.info("AURORA_BAILIAN_EMBEDDING_API_KEY is configured")
        if ark_key:
            logger.info("AURORA_ARK_API_KEY is configured")
        return

    logger.info(
        "No embedding API key found. Using LocalSemanticEmbedding. "
        "Set AURORA_EMBEDDING_PROVIDER to bailian, ark, local, or hash."
    )


def create_llm_provider(settings: AuroraSettings) -> LLMProvider:
    if settings.llm_provider == "bailian":
        if not settings.bailian_llm_api_key:
            raise ConfigurationError("Bailian LLM provider selected but AURORA_BAILIAN_LLM_API_KEY is not set")
        try:
            from aurora.integrations.llm.bailian import BailianLLM

            logger.info("Using Bailian LLM provider with model: %s", settings.bailian_llm_model)
            return BailianLLM(
                api_key=settings.bailian_llm_api_key,
                model=settings.bailian_llm_model,
                base_url=settings.bailian_llm_base_url,
                max_retries=settings.llm_max_retries,
                timeout=settings.llm_timeout,
            )
        except Exception as exc:
            raise ConfigurationError(f"Failed to create Bailian LLM provider: {exc}") from exc

    if settings.llm_provider == "ark":
        if not settings.ark_api_key:
            raise ConfigurationError("Ark LLM provider selected but AURORA_ARK_API_KEY is not set")
        try:
            from aurora.integrations.llm.ark import ArkLLM

            logger.info("Using Ark LLM provider with model: %s", settings.ark_llm_model)
            return ArkLLM(
                api_key=settings.ark_api_key,
                model=settings.ark_llm_model,
                base_url=settings.ark_base_url,
                max_retries=settings.llm_max_retries,
                timeout=settings.llm_timeout,
            )
        except Exception as exc:
            raise ConfigurationError(f"Failed to create Ark LLM provider: {exc}") from exc

    return MockLLM()


def create_embedding_provider(settings: AuroraSettings) -> EmbeddingProvider:
    provider = settings.embedding_provider

    if provider == "bailian":
        if not settings.bailian_embedding_api_key:
            raise ConfigurationError("Bailian embedding provider selected but AURORA_BAILIAN_EMBEDDING_API_KEY is not set")
        try:
            from aurora.integrations.embeddings.bailian import BailianEmbedding

            logger.info("Using Bailian embedding provider with model: %s", settings.bailian_embedding_model)
            return BailianEmbedding(
                api_key=settings.bailian_embedding_api_key,
                model=settings.bailian_embedding_model,
                base_url=settings.bailian_embedding_base_url,
                dimension=settings.dim,
                use_cache=settings.embedding_cache_enabled,
                cache_size=settings.embedding_cache_size,
            )
        except Exception as exc:
            raise ConfigurationError(f"Failed to create Bailian embedding provider: {exc}") from exc

    if provider == "ark":
        if not settings.ark_api_key:
            raise ConfigurationError("Ark embedding provider selected but AURORA_ARK_API_KEY is not set")
        try:
            from aurora.integrations.embeddings.ark import ArkEmbedding

            logger.info("Using Ark embedding provider")
            return ArkEmbedding(
                api_key=settings.ark_api_key,
                dimension=settings.dim,
                use_cache=settings.embedding_cache_enabled,
                cache_size=settings.embedding_cache_size,
            )
        except Exception as exc:
            raise ConfigurationError(f"Failed to create Ark embedding provider: {exc}") from exc

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


def create_memory(*, settings: AuroraSettings) -> AuroraMemory:
    cfg = build_memory_config(settings)
    embedder = create_embedding_provider(settings)
    return AuroraMemory(
        cfg=cfg,
        seed=int(settings.memory_seed),
        embedder=embedder,
        benchmark_mode=cfg.benchmark_mode,
    )
