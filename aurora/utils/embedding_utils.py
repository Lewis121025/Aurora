"""Embedding utility functions."""
from typing import Any, Optional

import numpy as np


def get_embedding_from_object(obj: Any) -> Optional[np.ndarray]:
    """Extract embedding from various object types.
    
    Tries to get embedding from 'embedding', 'centroid', or 'prototype' attributes.
    Handles list-to-ndarray conversion.
    
    Args:
        obj: Object that may have an embedding attribute.
        
    Returns:
        The embedding as numpy array, or None if not found.
    """
    for attr in ('embedding', 'centroid', 'prototype'):
        val = getattr(obj, attr, None)
        if val is not None:
            if isinstance(val, list):
                return np.array(val, dtype=np.float32)
            return val
    return None
