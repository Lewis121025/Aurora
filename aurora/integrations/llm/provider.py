"""
aurora/integrations/llm/provider.py
结构化多模态 LLM 提供者抽象。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Sequence, Type, TypeVar

from pydantic import BaseModel

from aurora.soul.models import Message

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """Message-part aware LLM interface."""

    @abstractmethod
    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout_s: float = 30.0,
        stop: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        max_retries: Optional[int] = None,
    ) -> Message:
        raise NotImplementedError

    def stream_complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout_s: float = 30.0,
        stop: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        max_retries: Optional[int] = None,
    ) -> Iterator[str]:
        reply = self.complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            stop=stop,
            metadata=metadata,
            max_retries=max_retries,
        )
        for part in reply.parts:
            text = getattr(part, "text", "")
            if text:
                yield str(text)

    @abstractmethod
    def complete_json(
        self,
        *,
        messages: Sequence[Message],
        schema: Type[T],
        temperature: float = 0.2,
        timeout_s: float = 30.0,
        metadata: Optional[Dict[str, Any]] = None,
        max_retries: Optional[int] = None,
    ) -> T:
        raise NotImplementedError
