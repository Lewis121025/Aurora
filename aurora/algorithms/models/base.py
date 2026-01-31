"""
AURORA Model Base Classes
=========================

Base classes and mixins for all AURORA data models.

Provides:
- SerializableMixin: Unified serialization interface for all models
- TimestampedMixin: Automatic timestamp management
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import fields, is_dataclass
from typing import Any, Dict, Type, TypeVar

import numpy as np


T = TypeVar("T")


class SerializableMixin(ABC):
    """
    Mixin providing unified serialization interface for dataclasses.
    
    All AURORA models should implement these methods consistently:
    - to_state_dict(): Convert to JSON-compatible dict
    - from_state_dict(): Reconstruct from state dict
    
    This ensures consistent serialization across all model types,
    enabling reliable persistence and state transfer.
    """
    
    @abstractmethod
    def to_state_dict(self) -> Dict[str, Any]:
        """
        Serialize the model to a JSON-compatible dictionary.
        
        Implementation notes:
        - np.ndarray should be converted to list via .tolist()
        - Nested SerializableMixin objects should call their to_state_dict()
        - Optional fields that are None should be included as None
        
        Returns:
            JSON-compatible dictionary representing the model state
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_state_dict(cls: Type[T], d: Dict[str, Any]) -> T:
        """
        Reconstruct a model instance from a state dictionary.
        
        Implementation notes:
        - Lists should be converted to np.ndarray where appropriate
        - Nested objects should be reconstructed via their from_state_dict()
        - Missing keys should use sensible defaults
        
        Args:
            d: State dictionary from to_state_dict()
            
        Returns:
            Reconstructed model instance
        """
        pass


def serialize_value(value: Any) -> Any:
    """
    Helper to serialize a single value to JSON-compatible format.
    
    Handles:
    - np.ndarray -> list
    - SerializableMixin -> dict
    - Basic types -> as-is
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    elif hasattr(value, 'to_state_dict'):
        return value.to_state_dict()
    elif isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    elif isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    else:
        return value


def auto_serialize(obj: Any) -> Dict[str, Any]:
    """
    Automatically serialize a dataclass to a dict.
    
    Uses reflection to serialize all fields.
    Override specific fields by implementing custom to_state_dict().
    
    Args:
        obj: A dataclass instance
        
    Returns:
        JSON-compatible dictionary
    """
    if not is_dataclass(obj):
        raise TypeError(f"auto_serialize requires a dataclass, got {type(obj)}")
    
    result = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        result[f.name] = serialize_value(value)
    return result
