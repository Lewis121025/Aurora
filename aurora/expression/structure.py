"""结构驱动认知上下文模块。

简化版：不再使用复杂的线程和记忆结系统。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StructuralContext:
    """结构化认知上下文。"""

    hint: str = ""


def build_structural_context(
    user_text: str,
    relation_id: str,
    formation: None,
    memory_store: None,
) -> StructuralContext:
    """构建结构化上下文（简化版）。"""
    return StructuralContext(hint="")
