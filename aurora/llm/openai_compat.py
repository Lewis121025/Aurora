"""OpenAI-compatible LLM provider implementation.

Uses standard library urllib to implement an HTTP client compatible with OpenAI API,
supporting any OpenAI-compatible interface (e.g., Alibaba Cloud Bailian, DeepSeek, etc.).
"""
from __future__ import annotations

import json
from typing import Any

import urllib.request

from aurora.llm.config import LLMConfig


class OpenAICompatProvider:
    """OpenAI-compatible LLM provider.

    Calls LLM API via HTTP POST requests and returns response text.
    No third-party HTTP library dependencies, uses only standard library urllib.

    Attributes:
        _config: LLM configuration object.
    """

    def __init__(self, config: LLMConfig) -> None:
        """Initialize the provider.

        Args:
            config: LLM configuration object.
        """
        self._config = config

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Call LLM to complete the request.

        Sends a chat completion request to the LLM API, parses the response,
        and returns the text content.

        Args:
            messages: List of messages, each containing role and content fields.

        Returns:
            LLM response text; empty string if no valid response.

        Raises:
            urllib.error.URLError: Network request failed.
            json.JSONDecodeError: Response parsing failed.
        """
        payload_obj: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": self._config.max_tokens,
        }
        if temperature is not None:
            payload_obj["temperature"] = temperature
        if max_tokens is not None:
            payload_obj["max_tokens"] = max_tokens
        if self._config.enable_thinking is not None:
            payload_obj["enable_thinking"] = self._config.enable_thinking

        payload = json.dumps(payload_obj).encode()

        request = urllib.request.Request(
            f"{self._config.base_url.rstrip('/')}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._config.api_key}",
            },
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=self._config.timeout_s) as response:
            data: dict[str, Any] = json.loads(response.read())

        choices = data.get("choices", [])
        if not choices:
            return ""
        return str(choices[0].get("message", {}).get("content", ""))
