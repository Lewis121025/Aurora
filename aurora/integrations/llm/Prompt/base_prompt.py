"""
aurora/integrations/llm/Prompt/base_prompt.py
基础提示词工具模块：定义了通用的指令模板和渲染函数。
用于确保所有 LLM 调用都遵循统一的 JSON 输出规范。
"""

from __future__ import annotations

from aurora.integrations.llm.schemas import SCHEMA_VERSION


def instruction(model_name: str) -> str:
    """
    生成强制 JSON 输出的指令后缀。

    参数：
        model_name: 期望符合的 Pydantic 模型类名。
    """
    return (
        "你必须且只能输出有效的 JSON。不要包含 Markdown 标记。不要包含多余的键。"
        f"Schema 版本: {SCHEMA_VERSION}。输出必须符合 {model_name} 的结构。"
    )
