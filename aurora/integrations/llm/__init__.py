"""
AURORA LLM 提供者
====================

可用的提供者：
- MockLLM：本地测试，无需 API 调用
- ArkLLM：火山方舟生产提供者
"""

from aurora.integrations.llm.provider import LLMProvider
from aurora.integrations.llm.mock import MockLLM

__all__ = ["LLMProvider", "MockLLM"]

# 可选提供者的延迟导入
def get_ark_llm():
    """获取 Ark LLM 提供者（需要 openai 包）。"""
    from aurora.integrations.llm.ark import ArkLLM, ArkLLMWithFallback
    return ArkLLM, ArkLLMWithFallback
