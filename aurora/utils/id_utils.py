"""
AURORA ID Utilities
===================

Deterministic ID generation for event-sourcing and reproducibility.
All IDs are stable across restarts and replays.
"""

from __future__ import annotations

import hashlib
import uuid


# Namespace UUID for AURORA deterministic IDs
AURORA_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "aurora-memory")


def det_id(kind: str, seed: str) -> str:
    """
    Generate deterministic UUID (stable across restarts/replays).

    Args:
        kind: Type of entity (e.g., "plot", "story", "theme")
        seed: Unique seed string for this entity

    Returns:
        Deterministic UUID string

    Example:
        >>> det_id("plot", "user123:1234567890.123")
        'a1b2c3d4-e5f6-5789-abcd-ef0123456789'
    """
    return str(uuid.uuid5(AURORA_NAMESPACE, f"{kind}:{seed}"))


def stable_hash(text: str) -> int:
    """
    Deterministic hash across runs (unlike Python's built-in salted hash).

    Uses SHA256 truncated to 64 bits for performance while maintaining
    sufficient collision resistance for typical use cases.

    Args:
        text: String to hash

    Returns:
        64-bit integer hash value
    """
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def content_hash(content: str, prefix: str = "") -> str:
    """
    Generate a content-based hash ID.

    Useful for deduplication and content-addressable storage.

    Args:
        content: Content to hash
        prefix: Optional prefix for the ID

    Returns:
        Hash-based ID string
    """
    h = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}{h}" if prefix else h
