"""AURORA Custom Exceptions.

This module defines a hierarchy of custom exceptions for the AURORA memory system.
All exceptions inherit from AuroraError to allow broad catching when needed.
"""


class AuroraError(Exception):
    """Base exception for all AURORA errors."""

    pass


class MemoryNotFoundError(AuroraError):
    """Memory element (Plot, Story, Theme) not found.

    Args:
        kind: Type of memory element ('plot', 'story', 'theme')
        element_id: The ID that was not found
    """

    def __init__(self, kind: str, element_id: str):
        self.kind = kind
        self.element_id = element_id
        super().__init__(f"{kind} not found: {element_id}")


class ConfigurationError(AuroraError):
    """Invalid configuration."""

    pass


class SerializationError(AuroraError):
    """Serialization or deserialization failed."""

    pass


class EmbeddingError(AuroraError):
    """Embedding generation failed."""

    pass


class LLMError(AuroraError):
    """LLM call failed."""

    pass


class StorageError(AuroraError):
    """Storage operation failed."""

    pass


class ValidationError(AuroraError):
    """Input validation failed."""

    pass
