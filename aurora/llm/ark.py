"""
火山方舟 (Volcengine Ark) LLM Provider
======================================

Production-ready LLM provider for AURORA memory system.
Supports structured JSON output with retry and error handling.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

from .provider import LLMProvider

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


class ArkLLM(LLMProvider):
    """火山方舟 LLM provider with structured output support.
    
    Features:
    - JSON schema constrained outputs
    - Automatic retry with exponential backoff
    - Token usage tracking
    - Error handling and fallback
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "doubao-1-5-pro-32k-250115",
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
        max_retries: int = 3,
        timeout: float = 60.0,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None
        self._total_tokens_used = 0
        
    def _get_client(self):
        """Lazy initialization of OpenAI client (Ark uses OpenAI-compatible API)."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                raise ImportError(
                    "Please install openai package: pip install openai"
                )
        return self._client
    
    def _build_json_schema(self, schema: Type[T]) -> Dict[str, Any]:
        """Build JSON schema from Pydantic model for structured output."""
        json_schema = schema.model_json_schema()
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": json_schema,
                "strict": True,
            }
        }
    
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
        """Complete with structured JSON output.
        
        Args:
            system: System prompt
            user: User prompt
            schema: Pydantic model for response validation
            temperature: Sampling temperature
            timeout_s: Request timeout in seconds
            metadata: Optional metadata for logging
            
        Returns:
            Validated Pydantic model instance
        """
        client = self._get_client()
        
        # Build a cleaner schema description for the prompt
        json_schema = schema.model_json_schema()
        required_fields = json_schema.get("required", [])
        properties = json_schema.get("properties", {})
        
        # Create a simplified schema hint
        schema_hint = {
            k: v.get("type", v.get("anyOf", [{}])[0].get("type", "string")) 
            for k, v in properties.items()
        }
        
        # Enhance prompt with strict JSON instructions
        enhanced_system = (
            f"{system}\n\n"
            f"【重要】你必须严格按照以下JSON格式返回结果，不要添加任何额外的包装或解释：\n"
            f"必填字段: {required_fields}\n"
            f"格式示例: {json.dumps(schema_hint, ensure_ascii=False)}"
        )
        
        messages = [
            {"role": "system", "content": enhanced_system},
            {"role": "user", "content": user},
        ]
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    timeout=timeout_s,
                )
                
                content = response.choices[0].message.content
                
                # Track token usage
                if hasattr(response, 'usage') and response.usage:
                    self._total_tokens_used += response.usage.total_tokens
                
                # Parse JSON with multiple strategies
                data = self._parse_json_response(content, schema)
                
                return schema.model_validate(data)
                
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    sleep_time = (2 ** attempt) * 0.5
                    time.sleep(sleep_time)
        
        raise RuntimeError(f"All {self.max_retries} attempts failed. Last error: {last_error}")
    
    def _parse_json_response(self, content: str, schema: Type[T]) -> Dict[str, Any]:
        """Parse JSON from LLM response with multiple fallback strategies."""
        import re
        
        # Strategy 1: Direct parse
        try:
            data = json.loads(content)
            # Check if data is wrapped in schema name
            schema_name = schema.__name__
            if isinstance(data, dict):
                if schema_name in data:
                    return data[schema_name]
                # Check for common wrapper patterns
                for key in list(data.keys()):
                    if key.lower() == schema_name.lower():
                        return data[key]
                    # If only one key and it's a dict, unwrap it
                    if len(data) == 1 and isinstance(data[key], dict):
                        inner = data[key]
                        # Check if inner has required fields
                        required = schema.model_json_schema().get("required", [])
                        if any(r in inner for r in required):
                            return inner
                return data
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract from markdown code block
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if match:
            try:
                data = json.loads(match.group(1))
                return self._unwrap_if_needed(data, schema)
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Find JSON object in content
        match = re.search(r'\{[\s\S]*\}', content)
        if match:
            try:
                data = json.loads(match.group(0))
                return self._unwrap_if_needed(data, schema)
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Could not extract valid JSON from response: {content[:300]}")
    
    def _unwrap_if_needed(self, data: Dict[str, Any], schema: Type[T]) -> Dict[str, Any]:
        """Unwrap nested data if wrapped in schema name."""
        if not isinstance(data, dict):
            return data
        schema_name = schema.__name__
        if schema_name in data:
            return data[schema_name]
        for key in list(data.keys()):
            if key.lower() == schema_name.lower():
                return data[key]
        return data
    
    def complete_text(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout_s: float = 30.0,
    ) -> str:
        """Complete with plain text output.
        
        Useful for free-form generation like summaries and narratives.
        """
        client = self._get_client()
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout_s,
        )
        
        if hasattr(response, 'usage') and response.usage:
            self._total_tokens_used += response.usage.total_tokens
        
        return response.choices[0].message.content
    
    @property
    def total_tokens_used(self) -> int:
        """Total tokens used across all requests."""
        return self._total_tokens_used
    
    def reset_token_counter(self) -> None:
        """Reset the token counter."""
        self._total_tokens_used = 0


class ArkLLMWithFallback(LLMProvider):
    """Ark LLM with MockLLM fallback for graceful degradation."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "doubao-1-5-pro-32k-250115",
        **kwargs,
    ):
        self._primary: Optional[ArkLLM] = None
        self._fallback = None
        
        if api_key:
            try:
                self._primary = ArkLLM(api_key=api_key, model=model, **kwargs)
            except Exception as e:
                logger.warning(f"Failed to initialize ArkLLM: {e}")
        
        # Lazy import fallback
        from .mock import MockLLM
        self._fallback = MockLLM()
    
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
        if self._primary:
            try:
                return self._primary.complete_json(
                    system=system,
                    user=user,
                    schema=schema,
                    temperature=temperature,
                    timeout_s=timeout_s,
                    metadata=metadata,
                )
            except Exception as e:
                logger.warning(f"Primary LLM failed, using fallback: {e}")
        
        return self._fallback.complete_json(
            system=system,
            user=user,
            schema=schema,
            temperature=temperature,
            timeout_s=timeout_s,
            metadata=metadata,
        )
