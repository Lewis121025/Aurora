"""LLM 提供者协议模块。

定义 LLM 提供者的接口契约，使 Aurora 可以接入任意 LLM 后端。
"""
from __future__ import annotations

from typing import Protocol


class LLMProvider(Protocol):
    """LLM 提供者协议。

    定义 LLM 提供者的最小接口要求，
    任何实现此协议的类都可作为 Aurora 的 LLM 后端。

    Methods:
        complete: 调用 LLM 完成补全。
    """

    def complete(self, messages: list[dict[str, str]]) -> str:
        """调用 LLM 完成补全。

        Args:
            messages: 消息列表，每条消息包含 role 和 content 字段。

        Returns:
            LLM 响应文本。
        """
        ...
