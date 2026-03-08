"""
阿里云百炼 (Alibaba Cloud Bailian) LLM 提供者
=====================================================

通过百炼 OpenAI 兼容接口调用通义千问模型。
当前默认模型为 `qwen3.5-plus`，可通过运行时配置覆盖。
"""

from __future__ import annotations

from typing import Optional

from aurora.integrations.llm.ark import ArkLLM, ArkLLMWithFallback


class BailianLLM(ArkLLM):
    """百炼 LLM 提供者。

    实现沿用 ArkLLM 的 OpenAI 兼容调用逻辑，仅替换百炼的默认模型与 base URL。
    """

    def __init__(
        self,
        api_key: str,
        model: str = "qwen3.5-plus",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        max_retries: int = 3,
        timeout: float = 60.0,
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
        )


class BailianLLMWithFallback(ArkLLMWithFallback):
    """百炼 LLM 与 MockLLM 回退封装。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen3.5-plus",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        **kwargs,
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            **kwargs,
        )
