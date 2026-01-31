"""
AURORA Embedding Providers
==========================

Available providers:
- LocalSemanticEmbedding: Local semantic embedding using word vectors (recommended for testing)
- HashEmbedding: Local deterministic testing (no API calls, random vectors)
- BailianEmbedding: 阿里云百炼 production provider
- ArkEmbedding: 火山方舟 provider

Usage:
    # For testing with semantic similarity
    from aurora.embeddings import LocalSemanticEmbedding
    embedder = LocalSemanticEmbedding(dim=384)
    
    # For legacy testing (random vectors, no semantic meaning)
    from aurora.embeddings import HashEmbedding
    embedder = HashEmbedding(dim=384)
    
    # For production (lazy import to avoid dependency issues)
    BailianEmbedding, _ = get_bailian_embedding()
    embedder = BailianEmbedding(api_key="...")
"""

from aurora.embeddings.base import EmbeddingProvider
from aurora.embeddings.hash import HashEmbedding
from aurora.embeddings.local_semantic import LocalSemanticEmbedding

__all__ = [
    "EmbeddingProvider",
    "HashEmbedding",
    "LocalSemanticEmbedding",
]


def get_bailian_embedding():
    """Get Bailian embedding provider (requires openai package)."""
    from aurora.embeddings.bailian import BailianEmbedding, BailianEmbeddingWithFallback
    return BailianEmbedding, BailianEmbeddingWithFallback


def get_ark_embedding():
    """Get Ark embedding provider (requires volcengine SDK)."""
    from aurora.embeddings.ark import ArkEmbedding, ArkEmbeddingWithFallback
    return ArkEmbedding, ArkEmbeddingWithFallback
