from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """LLM provider interface.

    Production:
      - implement JSON schema constrained outputs (tool calling / response_format JSON)
      - enforce max tokens / retry / backoff / circuit breaker
    
    Provides two main methods:
      - complete(): For text generation
      - complete_json(): For structured JSON output with schema validation
    """

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout_s: float = 30.0,
        stop: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate text completion for the given prompt.
        
        Args:
            prompt: The user prompt to complete
            system: Optional system message
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            timeout_s: Request timeout in seconds
            stop: Optional stop sequences
            metadata: Optional request metadata
            
        Returns:
            Generated text string
        """
        raise NotImplementedError

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
        """Generate structured JSON output conforming to schema.
        
        Args:
            system: System message
            user: User message
            schema: Pydantic model class for response validation
            temperature: Sampling temperature (0.0-1.0)
            timeout_s: Request timeout in seconds
            metadata: Optional request metadata
            
        Returns:
            Parsed and validated response as schema instance
        """
        raise NotImplementedError
