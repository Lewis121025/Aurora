"""
AURORA 本地语义嵌入
===============================

使用词向量进行基本相似性的本地语义嵌入。
语义相似的文本会产生相似的向量（不同于 HashEmbedding）。

使用方法：
    from aurora.integrations.embeddings import LocalSemanticEmbedding

    embedder = LocalSemanticEmbedding(dim=384)
    vec1 = embedder.embed_text("我住在北京")
    vec2 = embedder.embed_text("我在北京生活")
    # vec1 和 vec2 会相似，因为它们共享词汇
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Sequence

import numpy as np

from aurora.integrations.embeddings.base import ContentEmbeddingProvider, TextEmbeddingProvider
from aurora.soul.models import Message, messages_have_image_parts, messages_text_only
from aurora.utils.math_utils import l2_normalize


class LocalSemanticEmbedding(ContentEmbeddingProvider, TextEmbeddingProvider):
    """本地语义嵌入，使用词频向量捕获基本语义相似性。

    比 HashEmbedding 更好，因为语义相似的文本会有相似的向量。

    原理：
    - 每个词映射到一个确定性的随机向量（基于词的哈希）
    - 文本的向量是其所有词向量的加权平均
    - 共享相同词的文本会有相似的向量

    Attributes:
        dim: Embedding dimension
        seed: Random seed for reproducibility
    """

    def __init__(self, dim: int = 384, seed: int = 0):
        """初始化本地语义嵌入。

        参数：
            dim: 嵌入维度（默认 384）
            seed: 用于可重复性的随机种子
        """
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # 缓存词向量以提高性能
        self._word_vectors: Dict[str, np.ndarray] = {}

    def _get_word_vector(self, word: str) -> np.ndarray:
        """获取词的固定投影向量（确定性）。

        使用词的 MD5 哈希生成确定性随机向量。
        相同的词总是映射到相同的向量。

        参数：
            word: 要获取向量的词

        返回：
            形状为 (dim,) 的归一化词向量
        """
        if word not in self._word_vectors:
            # 使用词的哈希 + seed 生成确定性向量
            word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16)
            word_rng = np.random.default_rng((word_hash ^ self.seed) % (2**32))
            vec = word_rng.normal(size=self.dim).astype(np.float32)
            # 归一化词向量
            self._word_vectors[word] = l2_normalize(vec)
        return self._word_vectors[word]

    def _tokenize(self, text: str) -> List[str]:
        """简单分词，支持中英文。

        英文：按空格分词，只保留字母
        中文：按字符分词（简单但有效）

        参数：
            text: 要分词的输入文本

        返回：
            词元列表
        """
        # 转小写
        text = text.lower()
        tokens = []

        # 英文按单词分词
        english_words = re.findall(r"[a-z]+", text)
        tokens.extend(english_words)

        # 中文按字符分词（对于简单语义相似度已足够）
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
        tokens.extend(chinese_chars)

        # 数字也可能有意义
        numbers = re.findall(r"\d+", text)
        tokens.extend(numbers)

        return tokens

    def embed_text(self, text: str) -> np.ndarray:
        """生成文本的语义向量。

        使用词袋模型：将所有词向量取平均。
        共享词的文本会有相似的向量。

        参数：
            text: 要嵌入的输入文本

        返回：
            形状为 (dim,) 的归一化嵌入向量
        """
        tokens = self._tokenize(text)

        if not tokens:
            # 空文本返回零向量
            return np.zeros(self.dim, dtype=np.float32)

        # 词频统计（TF）
        word_counts: Dict[str, int] = {}
        for token in tokens:
            word_counts[token] = word_counts.get(token, 0) + 1

        # 加权词向量平均
        total_count = sum(word_counts.values())
        embedding = np.zeros(self.dim, dtype=np.float32)

        for word, count in word_counts.items():
            weight = count / total_count  # TF 权重
            embedding += weight * self._get_word_vector(word)

        # L2 归一化
        return l2_normalize(embedding)

    def embed_text_batch(self, texts: Sequence[str]) -> List[np.ndarray]:
        """嵌入多个文本。

        参数：
            texts: 要嵌入的文本列表

        返回：
            嵌入向量列表
        """
        return [self.embed_text(text) for text in texts]

    def similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的余弦相似度。

        便捷方法，用于快速测试语义相似性。

        参数：
            text1: 第一个文本
            text2: 第二个文本

        返回：
            范围在 [-1, 1] 的余弦相似度
        """
        vec1 = self.embed_text(text1)
        vec2 = self.embed_text(text2)
        return float(np.dot(vec1, vec2))

    def embed_content(self, messages: Sequence[Message]) -> np.ndarray:
        if messages_have_image_parts(messages):
            raise ValueError("LocalSemanticEmbedding is text-only and cannot embed image parts")
        return self.embed_text(messages_text_only(messages))

    def embed_content_batch(self, items: Sequence[Sequence[Message]]) -> List[np.ndarray]:
        return [self.embed_content(messages) for messages in items]
