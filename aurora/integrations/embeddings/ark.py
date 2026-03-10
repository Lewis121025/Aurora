"""
火山方舟 (Volcengine Ark) 嵌入提供者
=============================================

AURORA 内存系统的生产就绪嵌入提供者。
通过 volcengine SDK 使用 Doubao 嵌入模型。
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Sequence

import numpy as np

from .base import EmbeddingProvider

logger = logging.getLogger(__name__)


class ArkEmbedding(EmbeddingProvider):
    """使用 Doubao 嵌入模型的火山方舟 嵌入提供者。

    特性：
    - 高质量的中英文嵌入
    - 批处理支持
    - 自动重试和指数退避
    - MRL（Matryoshka 表示学习）维度支持
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "doubao-embedding-large-text-250515",
        max_retries: int = 3,
        dimension: int = 1024,  # Doubao 嵌入维度（支持 256, 512, 1024, 2048）
        use_cache: bool = True,
        cache_size: int = 10000,
    ):
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.dimension = dimension
        self.use_cache = use_cache
        self._client = None
        self._cache = {} if use_cache else None
        self._cache_size = cache_size
        self._total_requests = 0
        self._cache_hits = 0
        
    def _get_client(self):
        """Ark 客户端的延迟初始化。"""
        if self._client is None:
            try:
                from volcenginesdkarkruntime import Ark
                self._client = Ark(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "请安装 volcengine SDK：pip install 'volcengine-python-sdk[ark]'"
                )
        return self._client
    
    def _cache_key(self, text: str) -> str:
        """从文本生成缓存键。"""
        import hashlib
        return hashlib.sha256(text.encode()).hexdigest()[:32]
    
    def _get_from_cache(self, text: str) -> Optional[List[float]]:
        """如果可用，从缓存获取嵌入。"""
        if not self.use_cache or self._cache is None:
            return None
        key = self._cache_key(text)
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]
        return None
    
    def _add_to_cache(self, text: str, embedding: List[float]) -> None:
        """将嵌入添加到缓存。"""
        if not self.use_cache or self._cache is None:
            return
        # 类似 LRU 的驱逐：当满时移除最旧的条目
        if len(self._cache) >= self._cache_size:
            # 移除前 10% 的条目
            keys_to_remove = list(self._cache.keys())[:self._cache_size // 10]
            for key in keys_to_remove:
                del self._cache[key]
        key = self._cache_key(text)
        self._cache[key] = embedding
    
    def embed(self, text: str) -> np.ndarray:
        """为单个文本生成嵌入。

        参数：
            text: 要嵌入的输入文本

        返回：
            表示嵌入向量的浮点数列表
        """
        # 先检查缓存
        cached = self._get_from_cache(text)
        if cached is not None:
            return np.array(cached, dtype=np.float32)

        self._total_requests += 1
        client = self._get_client()

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = client.embeddings.create(
                    model=self.model,
                    input=[text],
                    encoding_format="float",
                )

                # 获取嵌入并截断到指定维度
                full_embedding = response.data[0].embedding
                embedding = np.array(full_embedding[:self.dimension], dtype=np.float32)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

                # 添加到缓存
                self._add_to_cache(text, embedding.tolist())

                return embedding
                
            except Exception as e:
                last_error = e
                logger.warning(f"Embedding attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    sleep_time = (2 ** attempt) * 0.3
                    time.sleep(sleep_time)
        
        raise RuntimeError(f"All {self.max_retries} embedding attempts failed. Last error: {last_error}")
    
    def embed_batch(self, texts: Sequence[str]) -> List[np.ndarray]:
        """为多个文本高效生成嵌入。

        尽可能使用批处理 API，对重复文本进行缓存。

        参数：
            texts: 输入文本列表

        返回：
            嵌入向量列表
        """
        if not texts:
            return []

        # 检查所有文本的缓存
        results: List[Optional[np.ndarray]] = [None] * len(texts)
        texts_to_embed = []
        indices_to_embed = []

        for i, text in enumerate(texts):
            cached = self._get_from_cache(text)
            if cached is not None:
                results[i] = cached
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)

        # 分批嵌入剩余文本
        if texts_to_embed:
            self._total_requests += 1
            client = self._get_client()

            # Ark API 支持批处理嵌入（推荐批大小 <= 4）
            batch_size = 4
            for batch_start in range(0, len(texts_to_embed), batch_size):
                batch_end = min(batch_start + batch_size, len(texts_to_embed))
                batch_texts = texts_to_embed[batch_start:batch_end]
                batch_indices = indices_to_embed[batch_start:batch_end]

                last_error = None
                for attempt in range(self.max_retries):
                    try:
                        response = client.embeddings.create(
                            model=self.model,
                            input=batch_texts,
                            encoding_format="float",
                        )

                        for j, data in enumerate(response.data):
                            idx = batch_indices[j]
                            full_embedding = data.embedding
                            embedding = np.array(full_embedding[:self.dimension], dtype=np.float32)
                            norm = np.linalg.norm(embedding)
                            if norm > 0:
                                embedding = embedding / norm

                            results[idx] = embedding
                            self._add_to_cache(texts[idx], embedding.tolist())

                        break

                    except Exception as e:
                        last_error = e
                        logger.warning(f"批处理嵌入尝试 {attempt + 1}/{self.max_retries} 失败：{e}")
                        if attempt < self.max_retries - 1:
                            sleep_time = (2 ** attempt) * 0.3
                            time.sleep(sleep_time)
                else:
                    # 所有重试都失败，回退到单个嵌入
                    logger.warning(f"批处理嵌入失败，回退到单个嵌入：{last_error}")
                    for j, text in enumerate(batch_texts):
                        idx = batch_indices[j]
                        try:
                            results[idx] = self.embed(text)
                        except Exception as e:
                            logger.error(f"单个嵌入也失败了：{e}")
                            # 作为最后手段返回零向量
                            results[idx] = np.zeros(self.dimension, dtype=np.float32)

        return results
    
    def embed_numpy(self, text: str) -> np.ndarray:
        """生成 numpy 数组形式的嵌入。

        便于 AURORA 的内部向量操作。
        """
        return self.embed(text)
    
    def embed_batch_numpy(self, texts: Sequence[str]) -> np.ndarray:
        """生成批量嵌入作为 numpy 数组。"""
        embeddings = self.embed_batch(texts)
        return np.stack(embeddings) if embeddings else np.array([], dtype=np.float32)
    
    @property
    def stats(self) -> dict:
        """返回缓存和请求统计信息。"""
        return {
            "total_requests": self._total_requests,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._cache) if self._cache else 0,
            "cache_hit_rate": self._cache_hits / max(1, self._total_requests + self._cache_hits),
        }
    
    def clear_cache(self) -> None:
        """清除嵌入缓存。"""
        if self._cache is not None:
            self._cache.clear()
