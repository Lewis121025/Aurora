"""表达上下文模块。

定义 LLM 认知过程的输入上下文结构：
- 用户输入文本
- 关系状态（RelationalState）
- 张力队列（TensionQueue）
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExpressionContext:
    """LLM 认知表达上下文。

    封装单次认知调用所需的全部上下文信息。

    Attributes:
        input_text: 用户输入文本。
        recalled_surfaces: 冷事实 + 会话记忆的表面描述。
        recent_summaries: 最近交互摘要。
        relational_state_segment: 关系状态投影（全量挂载）。
        tension_queue_segment: 张力队列投影。
    """

    input_text: str
    recalled_surfaces: tuple[str, ...] = ()
    recent_summaries: tuple[str, ...] = ()
    relational_state_segment: str = ""
    tension_queue_segment: str = ""
