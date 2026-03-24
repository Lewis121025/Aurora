"""OpenAI-compatible LLM provider implementation.

Uses standard library urllib to implement an HTTP client compatible with OpenAI API,
supporting any OpenAI-compatible interface (e.g., Alibaba Cloud Bailian, DeepSeek, etc.).
"""
from __future__ import annotations

import json
from typing import Any

import urllib.error
import urllib.request

from aurora.llm.config import LLMConfig


class OpenAICompatError(RuntimeError):
    """Base error for OpenAI-compatible provider failures."""


class OpenAICompatProtocolError(OpenAICompatError):
    """Raised when the upstream response shape is invalid."""


class OpenAICompatRefusalError(OpenAICompatError):
    """Raised when the model returns an explicit refusal."""


def _excerpt(value: object, *, limit: int = 240) -> str:
    if isinstance(value, (dict, list)):
        raw = json.dumps(value, ensure_ascii=False)
    else:
        raw = str(value)
    if len(raw) <= limit:
        return raw
    return f"{raw[:limit]}..."


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

        try:
            with urllib.request.urlopen(request, timeout=self._config.timeout_s) as response:
                data = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace").strip()
            detail = body or str(exc.reason or "empty response body")
            raise OpenAICompatError(f"LLM request failed with HTTP {exc.code}: {_excerpt(detail)}") from exc
        except json.JSONDecodeError as exc:
            raise OpenAICompatProtocolError("LLM response body is not valid JSON") from exc

        if not isinstance(data, dict):
            raise OpenAICompatProtocolError("LLM response must be a JSON object")

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise OpenAICompatProtocolError(
                f"LLM response is missing choices: {_excerpt(data)}"
            )

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise OpenAICompatProtocolError("LLM response choice must be an object")

        finish_reason = first_choice.get("finish_reason")
        if finish_reason == "length":
            raise OpenAICompatProtocolError("LLM response was truncated with finish_reason=length")
        if finish_reason == "content_filter":
            raise OpenAICompatProtocolError("LLM response was halted by content_filter")

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise OpenAICompatProtocolError("LLM response choice is missing message")

        refusal = message.get("refusal")
        if isinstance(refusal, str) and refusal.strip():
            raise OpenAICompatRefusalError(f"LLM refusal: {refusal.strip()}")
        if refusal not in {None, ""}:
            raise OpenAICompatProtocolError(f"LLM refusal has unsupported shape: {_excerpt(refusal)}")

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise OpenAICompatProtocolError(
                f"LLM response is missing message.content: {_excerpt(message)}"
            )
        return content.strip()
