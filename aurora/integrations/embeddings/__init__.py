"""
AURORA 嵌入提供者
==========================

可用的提供者：
- LocalSemanticEmbedding: 使用词向量的本地语义嵌入（推荐用于测试）
- HashEmbedding: 本地确定性测试（无API调用，随机向量）
- BailianEmbedding: 阿里云百炼 生产提供者
- ArkEmbedding: 火山方舟 提供者

使用方法：
    # 用于语义相似性测试
    from aurora.integrations.embeddings import LocalSemanticEmbedding
    embedder = LocalSemanticEmbedding(dim=384)

    # 用于遗留测试（随机向量，无语义含义）
    from aurora.integrations.embeddings import HashEmbedding
    embedder = HashEmbedding(dim=384)

    # 用于生产环境（延迟导入以避免依赖问题）
    BailianEmbedding, _ = get_bailian_embedding()
    embedder = BailianEmbedding(api_key="...")
"""

from aurora.integrations.embeddings.base import EmbeddingProvider
from aurora.integrations.embeddings.hash import HashEmbedding
from aurora.integrations.embeddings.local_semantic import LocalSemanticEmbedding

__all__ = [
    "EmbeddingProvider",
    "HashEmbedding",
    "LocalSemanticEmbedding",
]


def get_bailian_embedding():
    """获取 Bailian 嵌入提供者（需要 openai 包）。"""
    from aurora.integrations.embeddings.bailian import BailianEmbedding, BailianEmbeddingWithFallback
    return BailianEmbedding, BailianEmbeddingWithFallback


def get_ark_embedding():
    """获取 Ark 嵌入提供者（需要 volcengine SDK）。"""
    from aurora.integrations.embeddings.ark import ArkEmbedding, ArkEmbeddingWithFallback
    return ArkEmbedding, ArkEmbeddingWithFallback
