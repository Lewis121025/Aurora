"""
Alibaba Cloud Bailian embedding providers for Aurora V6.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, List, Optional, Sequence
from urllib import request

import numpy as np

from aurora.integrations.assets import AssetResolver
from aurora.soul.models import ImagePart, Message, TextPart, messages_to_text
from aurora.utils.math_utils import l2_normalize

from .base import ContentEmbeddingProvider, TextEmbeddingProvider

logger = logging.getLogger(__name__)


class _CacheMixin:
    def _cache_key(self, value: str) -> str:
        import hashlib

        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]

    def _get_from_cache(self, value: str) -> Optional[List[float]]:
        if not self.use_cache or self._cache is None:
            return None
        key = self._cache_key(value)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache_hits += 1
        return cached

    def _add_to_cache(self, value: str, embedding: List[float]) -> None:
        if not self.use_cache or self._cache is None:
            return
        if len(self._cache) >= self._cache_size:
            keys_to_remove = list(self._cache.keys())[: max(1, self._cache_size // 10)]
            for key in keys_to_remove:
                self._cache.pop(key, None)
        self._cache[self._cache_key(value)] = embedding


class BailianTextEmbedding(TextEmbeddingProvider, _CacheMixin):
    MAX_TEXT_LENGTH = 7500

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-v4",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        max_retries: int = 3,
        dimension: int = 1024,
        use_cache: bool = True,
        cache_size: int = 10000,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries
        self.dimension = dimension
        self.use_cache = use_cache
        self._cache: dict[str, list[float]] | None = {} if use_cache else None
        self._cache_size = cache_size
        self._client: Any | None = None
        self._total_requests = 0
        self._cache_hits = 0

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except ImportError as exc:  # pragma: no cover
                raise ImportError("请安装 openai 包：pip install openai") from exc
        return self._client

    def embed_text(self, text: str) -> np.ndarray:
        value = text[: self.MAX_TEXT_LENGTH]
        cached = self._get_from_cache(value)
        if cached is not None:
            return np.asarray(cached, dtype=np.float32)

        client = self._get_client()
        self._total_requests += 1
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = client.embeddings.create(
                    model=self.model,
                    input=value,
                    dimensions=self.dimension,
                )
                vector = np.asarray(response.data[0].embedding, dtype=np.float32)
                vector = l2_normalize(vector)
                self._add_to_cache(value, vector.tolist())
                return vector
            except Exception as exc:  # pragma: no cover - network failure path
                last_error = exc
                logger.warning("Bailian text embedding attempt %s failed: %s", attempt + 1, exc)
                if attempt < self.max_retries - 1:
                    time.sleep((2**attempt) * 0.3)
        raise RuntimeError(
            f"All {self.max_retries} text embedding attempts failed. Last error: {last_error}"
        )

    def embed_text_batch(self, texts: Sequence[str]) -> List[np.ndarray]:
        return [self.embed_text(text) for text in texts]


class BailianMultimodalEmbedding(ContentEmbeddingProvider, _CacheMixin):
    """DashScope multimodal embedding provider backed by the official multimodal endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str = "qwen-vl-embedding",
        base_url: str = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding",
        max_retries: int = 3,
        dimension: Optional[int] = None,
        use_cache: bool = True,
        cache_size: int = 10000,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries
        self.dimension = dimension
        self.use_cache = use_cache
        self._cache: dict[str, list[float]] | None = {} if use_cache else None
        self._cache_size = cache_size
        self._cache_hits = 0
        self._total_requests = 0
        self._assets = AssetResolver()

    def _payload_key(self, messages: Sequence[Message]) -> str:
        return messages_to_text(messages, include_image_uris=True)

    def _build_input(self, messages: Sequence[Message]) -> List[dict[str, str]]:
        items: List[dict[str, str]] = []
        for message in messages:
            for part in message.parts:
                if isinstance(part, TextPart):
                    if part.text.strip():
                        items.append({"text": part.text})
                    continue
                if isinstance(part, ImagePart):
                    items.append(
                        {
                            "image": self._assets.resolve_for_remote(
                                part.uri,
                                mime_type=part.mime_type,
                            )
                        }
                    )
                    continue
                raise TypeError(f"Unsupported message part: {type(part)!r}")
        if not items:
            raise ValueError("Cannot embed empty multimodal content")
        return items

    def _request_embedding(self, payload: dict[str, Any]) -> np.ndarray:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.base_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=60) as response:  # noqa: S310
            data = json.loads(response.read().decode("utf-8"))
        output = data.get("output", {})
        vector: Any = output.get("embedding")
        if vector is None:
            embeddings = output.get("embeddings")
            if isinstance(embeddings, list) and embeddings:
                first = embeddings[0]
                if isinstance(first, dict):
                    vector = first.get("embedding")
                else:
                    vector = first
        if not isinstance(vector, list):
            raise RuntimeError(f"Unexpected multimodal embedding response: {data}")
        array = np.asarray(vector, dtype=np.float32)
        if self.dimension is not None and self.dimension > 0:
            array = array[: self.dimension]
        return l2_normalize(array)

    def embed_content(self, messages: Sequence[Message]) -> np.ndarray:
        cache_value = self._payload_key(messages)
        cached = self._get_from_cache(cache_value)
        if cached is not None:
            return np.asarray(cached, dtype=np.float32)

        payload = {"model": self.model, "input": self._build_input(messages)}
        self._total_requests += 1
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                vector = self._request_embedding(payload)
                self._add_to_cache(cache_value, vector.tolist())
                return vector
            except Exception as exc:  # pragma: no cover - network failure path
                last_error = exc
                logger.warning(
                    "Bailian multimodal embedding attempt %s failed: %s",
                    attempt + 1,
                    exc,
                )
                if attempt < self.max_retries - 1:
                    time.sleep((2**attempt) * 0.3)
        raise RuntimeError(
            f"All {self.max_retries} multimodal embedding attempts failed. Last error: {last_error}"
        )

    def embed_content_batch(self, items: Sequence[Sequence[Message]]) -> List[np.ndarray]:
        return [self.embed_content(messages) for messages in items]
