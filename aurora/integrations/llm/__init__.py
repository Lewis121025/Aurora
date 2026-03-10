"""
aurora.integrations.llm
大语言模型 (LLM) 集成包。
本包提供了统一的 LLM 访问接口，并实现了对接火山方舟 (Ark)、阿里云百炼 (Bailian) 等国内主流厂商的提供者。
"""

from .provider import LLMProvider
from .schemas import *

__all__ = ["LLMProvider"]
