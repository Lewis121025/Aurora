"""
阿里云百炼 (Alibaba Cloud Bailian) 嵌入提供者
=====================================================

使用阿里云百炼的生产就绪嵌入提供者。
使用 OpenAI 兼容 API 进行文本嵌入。
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Sequence

import numpy as np

from .base import EmbeddingProvider

logger = logging.getLogger(__name__)


class BailianEmbedding(EmbeddingProvider):
    """阿里云百炼 嵌入提供者。

    特性：
    - 高质量的中英文嵌入
    - OpenAI 兼容 API
    - 批处理支持
    - 自动重试和指数退避
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-v3",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        max_retries: int = 3,
        dimension: int = 1024,
        use_cache: bool = True,
        cache_size: int = 10000,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries
        self.dimension = dimension
        self.use_cache = use_cache
        self._client = None
        self._cache = {} if use_cache else None
        self._cache_size = cache_size
        self._total_requests = 0
        self._cache_hits = 0
        
    def _get_client(self):
        """OpenAI 客户端的延迟初始化（百炼使用 OpenAI 兼容 API）。"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                raise ImportError(
                    "请安装 openai 包：pip install openai"
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
        if len(self._cache) >= self._cache_size:
            keys_to_remove = list(self._cache.keys())[:self._cache_size // 10]
            for key in keys_to_remove:
                del self._cache[key]
        key = self._cache_key(text)
        self._cache[key] = embedding
    
    # 百炼 API 有 8192 字符限制，使用 7500 以确保安全
    MAX_TEXT_LENGTH = 7500
    
    def embed(self, text: str) -> np.ndarray:
        """为单个文本生成嵌入。

        返回：
            形状为 (dimension,) 的 np.ndarray，dtype 为 float32，L2 归一化
        """
        # 如果文本过长则截断（API 限制：8192 字符）
        if len(text) > self.MAX_TEXT_LENGTH:
            text = text[:self.MAX_TEXT_LENGTH]
        
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
                    input=text,
                    dimensions=self.dimension,
                )
                embedding = response.data[0].embedding

                # 转换为 numpy 并归一化
                emb_array = np.array(embedding, dtype=np.float32)
                norm = np.linalg.norm(emb_array)
                if norm > 0:
                    emb_array = emb_array / norm

                # 作为列表缓存以提高存储效率
                self._add_to_cache(text, emb_array.tolist())
                return emb_array
                
            except Exception as e:
                last_error = e
                logger.warning(f"Embedding attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    sleep_time = (2 ** attempt) * 0.3
                    time.sleep(sleep_time)
        
        raise RuntimeError(f"All {self.max_retries} embedding attempts failed. Last error: {last_error}")
    
    def embed_batch(self, texts: Sequence[str]) -> List[np.ndarray]:
        """为多个文本生成嵌入。

        返回：
            np.ndarray 嵌入列表，每个形状为 (dimension,)，dtype 为 float32
        """
        if not texts:
            return []

        results: List[Optional[np.ndarray]] = [None] * len(texts)
        texts_to_embed = []
        indices_to_embed = []

        for i, text in enumerate(texts):
            # 如果文本过长则截断
            if len(text) > self.MAX_TEXT_LENGTH:
                text = text[:self.MAX_TEXT_LENGTH]
            cached = self._get_from_cache(text)
            if cached is not None:
                results[i] = np.array(cached, dtype=np.float32)
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
        
        if texts_to_embed:
            self._total_requests += 1
            client = self._get_client()
            
            batch_size = 10
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
                            dimensions=self.dimension,
                        )
                        
                        for j, data in enumerate(response.data):
                            idx = batch_indices[j]
                            emb_array = np.array(data.embedding, dtype=np.float32)
                            norm = np.linalg.norm(emb_array)
                            if norm > 0:
                                emb_array = emb_array / norm
                            results[idx] = emb_array
                            self._add_to_cache(texts[idx], emb_array.tolist())
                        break
                        
                    except Exception as e:
                        last_error = e
                        logger.warning(f"Batch embedding attempt {attempt + 1}/{self.max_retries} failed: {e}")
                        if attempt < self.max_retries - 1:
                            time.sleep((2 ** attempt) * 0.3)
                else:
                    for j, text in enumerate(batch_texts):
                        idx = batch_indices[j]
                        try:
                            results[idx] = self.embed(text)
                        except Exception as e:
                            logger.debug(f"文本索引 {idx} 的嵌入失败，使用零向量：{e}")
                            results[idx] = np.zeros(self.dimension, dtype=np.float32)
        
        return results
    
    def embed_batch_numpy(self, texts: Sequence[str]) -> np.ndarray:
        """生成批量嵌入作为堆叠的 numpy 数组。

        返回：
            形状为 (len(texts), dimension) 的 np.ndarray，dtype 为 float32
        """
        embeddings = self.embed_batch(texts)
        return np.stack(embeddings) if embeddings else np.array([], dtype=np.float32)
    
    @property
    def stats(self) -> dict:
        return {
            "total_requests": self._total_requests,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._cache) if self._cache else 0,
        }
    
    def clear_cache(self) -> None:
        if self._cache is not None:
            self._cache.clear()
