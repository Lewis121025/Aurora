from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """LLM provider interface.

    Production:
      - implement JSON schema constrained outputs (tool calling / response_format JSON)
      - enforce max tokens / retry / backoff / circuit breaker
    """

    @abstractmethod
    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: Type[T],
        temperature: float = 0.2,
        timeout_s: float = 30.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> T:
        raise NotImplementedError
