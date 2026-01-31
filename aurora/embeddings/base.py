from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence, Union

import numpy as np


class EmbeddingProvider(ABC):
    """Base class for embedding providers.
    
    All embedding providers should return numpy arrays for compatibility
    with the memory system's vector operations.
    """
    
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            np.ndarray of shape (dim,) with float32 dtype
        """
        raise NotImplementedError

    def embed_batch(self, texts: Sequence[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: Sequence of texts to embed
            
        Returns:
            List of np.ndarray embeddings
        """
        return [self.embed(t) for t in texts]
