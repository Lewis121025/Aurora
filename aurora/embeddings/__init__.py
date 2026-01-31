"""
AURORA Embedding Providers
==========================

Available providers:
- HashEmbedding: Local deterministic testing (no API calls)
- BailianEmbedding: 阿里云百炼 production provider
- ArkEmbedding: 火山方舟 provider

Usage:
    # For testing
    from aurora.embeddings import HashEmbedding
    embedder = HashEmbedding(dim=384)
    
    # For production (lazy import to avoid dependency issues)
    BailianEmbedding, _ = get_bailian_embedding()
    embedder = BailianEmbedding(api_key="...")
"""

from aurora.embeddings.base import EmbeddingProvider
from aurora.embeddings.hash import HashEmbedding

__all__ = [
    "EmbeddingProvider",
    "HashEmbedding",
]


def get_bailian_embedding():
    """Get Bailian embedding provider (requires openai package)."""
    from aurora.embeddings.bailian import BailianEmbedding, BailianEmbeddingWithFallback
    return BailianEmbedding, BailianEmbeddingWithFallback


def get_ark_embedding():
    """Get Ark embedding provider (requires volcengine SDK)."""
    from aurora.embeddings.ark import ArkEmbedding, ArkEmbeddingWithFallback
    return ArkEmbedding, ArkEmbeddingWithFallback
