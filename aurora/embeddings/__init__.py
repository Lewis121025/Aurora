"""
AURORA Embedding Providers
==========================

Available providers:
- HashEmbedding: Local testing without API calls
- BailianEmbedding: 阿里云百炼 (Alibaba Bailian) production provider
- ArkEmbedding: 火山方舟 (Volcengine Ark) provider
"""

from aurora.embeddings.base import EmbeddingProvider
from aurora.embeddings.hash import HashEmbedding

__all__ = ["EmbeddingProvider", "HashEmbedding"]

# Lazy imports for optional providers
def get_bailian_embedding():
    """Get Bailian embedding provider (requires openai package)."""
    from aurora.embeddings.bailian import BailianEmbedding, BailianEmbeddingWithFallback
    return BailianEmbedding, BailianEmbeddingWithFallback

def get_ark_embedding():
    """Get Ark embedding provider (requires volcengine SDK)."""
    from aurora.embeddings.ark import ArkEmbedding, ArkEmbeddingWithFallback
    return ArkEmbedding, ArkEmbeddingWithFallback
