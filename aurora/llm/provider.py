"""LLM provider protocol module.

Defines the interface contract for LLM providers, enabling Aurora to integrate
with any LLM backend.
"""
from __future__ import annotations

from typing import Protocol


class LLMProvider(Protocol):
    """LLM provider protocol.

    Defines the minimum interface requirements for LLM providers.
    Any class implementing this protocol can serve as Aurora's LLM backend.

    Methods:
        complete: Call LLM to complete the request.
    """

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Call LLM to complete the request.

        Args:
            messages: List of messages, each containing role and content fields.

        Returns:
            LLM response text.
        """
        ...
