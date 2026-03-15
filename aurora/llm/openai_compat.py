from __future__ import annotations

import json
import urllib.request
from typing import Any

from aurora.llm.config import LLMConfig


class OpenAICompatProvider:
    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    def complete(self, messages: list[dict[str, str]]) -> str:
        payload = json.dumps({
            "model": self._config.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 256,
        }).encode()
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


def create_provider(config: LLMConfig) -> OpenAICompatProvider:
    return OpenAICompatProvider(config)
