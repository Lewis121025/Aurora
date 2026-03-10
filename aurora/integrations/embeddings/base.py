from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np


class EmbeddingProvider(ABC):
    """嵌入提供者的基类。

    所有嵌入提供者应返回 numpy 数组，以与
    内存系统的向量操作兼容。
    """

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """为单个文本生成嵌入。

        参数：
            text: 要嵌入的文本

        返回：
            形状为 (dim,) 的 np.ndarray，dtype 为 float32
        """
        raise NotImplementedError

    def embed_batch(self, texts: Sequence[str]) -> List[np.ndarray]:
        """为多个文本生成嵌入。

        参数：
            texts: 要嵌入的文本序列

        返回：
            np.ndarray 嵌入列表
        """
        return [self.embed(t) for t in texts]
