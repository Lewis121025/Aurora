"""
aurora/integrations/llm/bailian.py
阿里云百炼 (Alibaba Cloud Bailian) 大语言模型提供者模块。
本模块通过百炼提供的 OpenAI 兼容接口，实现了对通义千问（Qwen）系列模型的调用。
核心实现复用了 ArkLLM 的 OpenAI 协议封装逻辑。
"""

from __future__ import annotations

from typing import Any, Dict

from aurora.integrations.llm.ark import ArkLLM


class BailianLLM(ArkLLM):
    """
    百炼 LLM 提供者实现类。
    
    主要特性：
    - 兼容 OpenAI Chat Completion API 规范。
    - 默认模型设为 `qwen3.5-plus`。
    - 自动处理重试与超时逻辑。
    """

    def __init__(
        self,
        api_key: str,
        model: str = "qwen3.5-plus",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        max_retries: int = 3,
        timeout: float = 60.0,
    ):
        """
        初始化百炼提供者。
        
        参数：
            api_key：阿里云百炼控制台获取的 API 密钥。
            model：要调用的模型标识，默认为 qwen3.5-plus。
            base_url：百炼 OpenAI 兼容接口地址。
            max_retries：请求失败时的最大重试次数。
            timeout：请求超时时间（秒）。
        """
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
        )

    def _request_options(self, *, structured: bool) -> Dict[str, Any]:
        """
        生成百炼特定的请求额外参数。
        
        对于百炼，我们显式禁用思维链（thinking）输出，以确保返回内容更符合系统处理逻辑。
        """
        return {"extra_body": {"enable_thinking": False}}
