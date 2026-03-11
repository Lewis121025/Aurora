"""
AURORA 哈希嵌入
=====================

用于测试的确定性伪随机嵌入。
基于文本哈希生成一致的嵌入。

使用方法：
    from aurora.integrations.embeddings import HashEmbedding

    embedder = HashEmbedding(dim=384)
    vec = embedder.embed_text("Hello world")
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from aurora.integrations.embeddings.base import ContentEmbeddingProvider, TextEmbeddingProvider
from aurora.soul.models import Message, messages_to_text
from aurora.utils.id_utils import stable_hash
from aurora.utils.math_utils import l2_normalize


class HashEmbedding(ContentEmbeddingProvider, TextEmbeddingProvider):
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

    def _hash_to_vector(self, value: str) -> np.ndarray:
        rng = np.random.default_rng(stable_hash(value) ^ self.seed)
        v = rng.normal(size=self.dim).astype(np.float32)
        return l2_normalize(v)

    def embed_text(self, text: str) -> np.ndarray:
        """从文本生成确定性嵌入。"""
        return self._hash_to_vector(text)

    def embed_text_batch(self, texts: Sequence[str]) -> List[np.ndarray]:
        """嵌入多个文本。"""
        return [self.embed_text(text) for text in texts]

    def embed_content(self, messages: Sequence[Message]) -> np.ndarray:
        return self._hash_to_vector(messages_to_text(messages, include_image_uris=True))

    def embed_content_batch(self, items: Sequence[Sequence[Message]]) -> List[np.ndarray]:
        return [self.embed_content(messages) for messages in items]
