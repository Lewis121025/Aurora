"""
DEPRECATED: HashEmbedding has moved to aurora.embeddings.hash

This module is kept for backward compatibility only.
Please update your imports:

    # Old (deprecated)
    from aurora.algorithms.components.embedding import HashEmbedding
    
    # New (preferred)
    from aurora.embeddings import HashEmbedding
"""

import warnings

warnings.warn(
    "aurora.algorithms.components.embedding is deprecated. "
    "Import from aurora.embeddings instead: "
    "from aurora.embeddings import HashEmbedding",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export for backward compatibility
from aurora.embeddings.hash import HashEmbedding

__all__ = ["HashEmbedding"]
