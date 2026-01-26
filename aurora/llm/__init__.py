"""
AURORA LLM Providers
====================

Available providers:
- MockLLM: Local testing without API calls
- ArkLLM: 火山方舟 (Volcengine Ark) production provider
"""

from aurora.llm.provider import LLMProvider
from aurora.llm.mock import MockLLM

__all__ = ["LLMProvider", "MockLLM"]

# Lazy imports for optional providers
def get_ark_llm():
    """Get Ark LLM provider (requires openai package)."""
    from aurora.llm.ark import ArkLLM, ArkLLMWithFallback
    return ArkLLM, ArkLLMWithFallback
