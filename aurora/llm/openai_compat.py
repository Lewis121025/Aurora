"""OpenAI 兼容 LLM 提供者实现。

使用标准库 urllib 实现与 OpenAI API 兼容的 HTTP 客户端，
支持任意 OpenAI 兼容接口（如阿里云百炼、DeepSeek 等）。
"""
from __future__ import annotations

import json
from typing import Any

import urllib.request

from aurora.llm.config import LLMConfig


class OpenAICompatProvider:
    """OpenAI 兼容 LLM 提供者。

    通过 HTTP POST 请求调用 LLM API，返回响应文本。
    不依赖第三方 HTTP 库，仅使用标准库 urllib。

    Attributes:
        _config: LLM 配置对象。
    """

    def __init__(self, config: LLMConfig) -> None:
        """初始化提供者。

        Args:
            config: LLM 配置对象。
        """
        self._config = config

    def complete(self, messages: list[dict[str, str]]) -> str:
        """调用 LLM 完成补全。

        发送聊天完成请求到 LLM API，解析响应并返回文本内容。

        Args:
            messages: 消息列表，每条消息包含 role 和 content 字段。

        Returns:
            LLM 响应文本；若无有效响应则返回空字符串。

        Raises:
            urllib.error.URLError: 网络请求失败。
            json.JSONDecodeError: 响应解析失败。
        """
        payload_obj: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": self._config.max_tokens,
        }
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
