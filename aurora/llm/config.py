"""LLM 配置模块。

定义 LLM 配置及运行时参数：
- LLM 配置（环境变量读取）
- 蒸馏阈值配置
- 会话超时配置
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """LLM 配置。

    Attributes:
        base_url: LLM API 基础 URL。
        model: 模型名称。
        api_key: API 密钥。
        timeout_s: 请求超时（秒）。
    """

    base_url: str
    model: str
    api_key: str
    timeout_s: float = 30.0
    max_tokens: int = 1024


DISTILL_THRESHOLD_TURNS = 20
"""蒸馏触发阈值：单会话累计轮数达到此值时触发蒸馏。"""

SESSION_IDLE_TIMEOUT_MINUTES = 30
"""会话空闲超时（分钟）：超过此时间无交互则触发蒸馏。"""

EMBEDDING_DIM = 384
"""向量维度（MiniLM all-MiniLM-L6-v2）。"""


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

    if not base_url or not api_key:
        return None

    return LLMConfig(
        base_url=base_url,
        model=model or "gpt-4o-mini",
        api_key=api_key,
        timeout_s=float(os.environ.get("AURORA_LLM_TIMEOUT_S", "30.0")),
        max_tokens=int(os.environ.get("AURORA_LLM_MAX_TOKENS", "1024")),
    )
