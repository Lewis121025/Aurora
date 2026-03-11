from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np

from aurora.soul.models import Message


class TextEmbeddingProvider(ABC):
    """Embed plain text for axis compilation, fact extraction, and text-only analysis."""

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def embed_text_batch(self, texts: Sequence[str]) -> List[np.ndarray]:
        return [self.embed_text(text) for text in texts]


class ContentEmbeddingProvider(ABC):
    """Embed structured multimodal message content into a shared vector space."""

    @abstractmethod
    def embed_content(self, messages: Sequence[Message]) -> np.ndarray:
        raise NotImplementedError

    def embed_content_batch(self, items: Sequence[Sequence[Message]]) -> List[np.ndarray]:
        return [self.embed_content(messages) for messages in items]
