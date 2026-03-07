"""
AURORA 哈希嵌入
=====================

用于测试的确定性伪随机嵌入。
基于文本哈希生成一致的嵌入。

使用方法：
    from aurora.embeddings import HashEmbedding

    embedder = HashEmbedding(dim=384)
    vec = embedder.embed("Hello world")
"""

from __future__ import annotations

from typing import List

import numpy as np

from aurora.embeddings.base import EmbeddingProvider
from aurora.utils.math_utils import l2_normalize
from aurora.utils.id_utils import stable_hash


class HashEmbedding(EmbeddingProvider):
    """用于测试的确定性伪随机嵌入。

    基于文本哈希生成一致的嵌入。
    在生产环境中应替换为真实的嵌入模型。

    属性：
        dim: 嵌入维度
        seed: 用于可重复性的随机种子
    """

    def __init__(self, dim: int = 384, seed: int = 7):
        self.dim = dim
        self.seed = seed

    def embed(self, text: str) -> np.ndarray:
        """从文本生成确定性嵌入。"""
        rng = np.random.default_rng(stable_hash(text) ^ self.seed)
        v = rng.normal(size=self.dim).astype(np.float32)
        return l2_normalize(v)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """嵌入多个文本。"""
        return [self.embed(t) for t in texts]
