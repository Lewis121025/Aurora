"""LLM 配置模块。

定义 LLM 配置结构及加载函数，从环境变量读取：
- AURORA_LLM_BASE_URL: LLM API 基础 URL
- AURORA_LLM_MODEL: 模型名称
- AURORA_LLM_API_KEY: API 密钥
- AURORA_LLM_TIMEOUT_S: 请求超时（秒）
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """LLM 配置。

    Attributes:
        base_url: LLM API 基础 URL（如 https://api.openai.com/v1）。
        model: 模型名称（如 gpt-4o-mini）。
        api_key: API 密钥。
        timeout_s: 请求超时（秒），默认 30 秒。
    """

    base_url: str
    model: str
    api_key: str
    timeout_s: float = 30.0


def load_llm_config() -> LLMConfig | None:
    """从环境变量加载 LLM 配置。

    必需环境变量：
        - AURORA_LLM_BASE_URL
        - AURORA_LLM_API_KEY

    可选环境变量：
        - AURORA_LLM_MODEL: 默认 "gpt-4o-mini"
        - AURORA_LLM_TIMEOUT_S: 默认 30.0

    Returns:
        LLMConfig: 配置对象；若必需变量缺失则返回 None。
    """
    base_url = os.environ.get("AURORA_LLM_BASE_URL", "")
    model = os.environ.get("AURORA_LLM_MODEL", "")
    api_key = os.environ.get("AURORA_LLM_API_KEY", "")

    # 必需变量缺失时返回 None
    if not base_url or not api_key:
        return None

    return LLMConfig(
        base_url=base_url,
        model=model or "gpt-4o-mini",
        api_key=api_key,
        timeout_s=float(os.environ.get("AURORA_LLM_TIMEOUT_S", "30.0")),
    )
