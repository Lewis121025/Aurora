"""
火山方舟 (Volcengine Ark) Embedding Provider
=============================================

Production-ready embedding provider for AURORA memory system.
Uses Doubao embedding model via volcengine SDK.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Sequence

import numpy as np

from .base import EmbeddingProvider

logger = logging.getLogger(__name__)


class ArkEmbedding(EmbeddingProvider):
    """火山方舟 embedding provider using Doubao embedding model.
    
    Features:
    - High-quality Chinese and English embeddings
    - Batch processing support
    - Automatic retry with exponential backoff
    - MRL (Matryoshka Representation Learning) dimension support
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "doubao-embedding-large-text-250515",
        max_retries: int = 3,
        dimension: int = 1024,  # Doubao embedding dimension (supports 256, 512, 1024, 2048)
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
        """Lazy initialization of Ark client."""
        if self._client is None:
            try:
                from volcenginesdkarkruntime import Ark
                self._client = Ark(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "Please install volcengine SDK: pip install 'volcengine-python-sdk[ark]'"
                )
        return self._client
    
    def _cache_key(self, text: str) -> str:
        """Generate cache key from text."""
        import hashlib
        return hashlib.sha256(text.encode()).hexdigest()[:32]
    
    def _get_from_cache(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache if available."""
        if not self.use_cache or self._cache is None:
            return None
        key = self._cache_key(text)
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]
        return None
    
    def _add_to_cache(self, text: str, embedding: List[float]) -> None:
        """Add embedding to cache."""
        if not self.use_cache or self._cache is None:
            return
        # Simple LRU-like eviction: remove oldest entries when full
        if len(self._cache) >= self._cache_size:
            # Remove first 10% of entries
            keys_to_remove = list(self._cache.keys())[:self._cache_size // 10]
            for key in keys_to_remove:
                del self._cache[key]
        key = self._cache_key(text)
        self._cache[key] = embedding
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        # Check cache first
        cached = self._get_from_cache(text)
        if cached is not None:
            return cached
        
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
                
                # Get embedding and truncate to specified dimension
                full_embedding = response.data[0].embedding
                embedding = full_embedding[:self.dimension]
                
                # Normalize the embedding
                norm = sum(x*x for x in embedding) ** 0.5
                if norm > 0:
                    embedding = [x / norm for x in embedding]
                
                # Add to cache
                self._add_to_cache(text, embedding)
                
                return embedding
                
            except Exception as e:
                last_error = e
                logger.warning(f"Embedding attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    sleep_time = (2 ** attempt) * 0.3
                    time.sleep(sleep_time)
        
        raise RuntimeError(f"All {self.max_retries} embedding attempts failed. Last error: {last_error}")
    
    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently.
        
        Uses batch API when possible, with caching for repeated texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Check cache for all texts
        results = [None] * len(texts)
        texts_to_embed = []
        indices_to_embed = []
        
        for i, text in enumerate(texts):
            cached = self._get_from_cache(text)
            if cached is not None:
                results[i] = cached
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
        
        # Embed remaining texts in batches
        if texts_to_embed:
            self._total_requests += 1
            client = self._get_client()
            
            # Ark API supports batch embedding (recommended batch size <= 4)
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
                            embedding = full_embedding[:self.dimension]
                            
                            # Normalize
                            norm = sum(x*x for x in embedding) ** 0.5
                            if norm > 0:
                                embedding = [x / norm for x in embedding]
                            
                            results[idx] = embedding
                            self._add_to_cache(texts[idx], embedding)
                        
                        break
                        
                    except Exception as e:
                        last_error = e
                        logger.warning(f"Batch embedding attempt {attempt + 1}/{self.max_retries} failed: {e}")
                        if attempt < self.max_retries - 1:
                            sleep_time = (2 ** attempt) * 0.3
                            time.sleep(sleep_time)
                else:
                    # All retries failed, fall back to individual embedding
                    logger.warning(f"Batch embedding failed, falling back to individual: {last_error}")
                    for j, text in enumerate(batch_texts):
                        idx = batch_indices[j]
                        try:
                            results[idx] = self.embed(text)
                        except Exception as e:
                            logger.error(f"Individual embedding also failed: {e}")
                            # Return zero vector as last resort
                            results[idx] = [0.0] * self.dimension
        
        return results
    
    def embed_numpy(self, text: str) -> np.ndarray:
        """Generate embedding as numpy array.
        
        Convenient for AURORA's internal vector operations.
        """
        return np.array(self.embed(text), dtype=np.float32)
    
    def embed_batch_numpy(self, texts: Sequence[str]) -> np.ndarray:
        """Generate batch embeddings as numpy array."""
        embeddings = self.embed_batch(texts)
        return np.array(embeddings, dtype=np.float32)
    
    @property
    def stats(self) -> dict:
        """Return cache and request statistics."""
        return {
            "total_requests": self._total_requests,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._cache) if self._cache else 0,
            "cache_hit_rate": self._cache_hits / max(1, self._total_requests + self._cache_hits),
        }
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if self._cache is not None:
            self._cache.clear()


class ArkEmbeddingWithFallback(EmbeddingProvider):
    """Ark embedding with hash embedding fallback for graceful degradation."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "doubao-embedding-large-text-250515",
        fallback_dim: int = 1024,
        **kwargs,
    ):
        self._primary: Optional[ArkEmbedding] = None
        self._fallback = None
        self.dimension = fallback_dim
        
        if api_key:
            try:
                self._primary = ArkEmbedding(api_key=api_key, model=model, dimension=fallback_dim, **kwargs)
                self.dimension = self._primary.dimension
            except Exception as e:
                logger.warning(f"Failed to initialize ArkEmbedding: {e}")
        
        # Lazy import fallback
        from .hash import HashEmbedding
        self._fallback = HashEmbedding(dim=fallback_dim)
    
    def embed(self, text: str) -> List[float]:
        if self._primary:
            try:
                return self._primary.embed(text)
            except Exception as e:
                logger.warning(f"Primary embedding failed, using fallback: {e}")
        
        return list(self._fallback.embed(text))
    
    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        if self._primary:
            try:
                return self._primary.embed_batch(texts)
            except Exception as e:
                logger.warning(f"Primary batch embedding failed, using fallback: {e}")
        
        return [list(self._fallback.embed(t)) for t in texts]
