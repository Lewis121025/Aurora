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
        dominant_channels: 主导通道列表（保留字段）。
        has_knots: 是否存在未解决悬案。
        recalled_surfaces: 回忆片段表面描述列表。
        recent_summaries: 最近交互摘要列表。
        relational_state_segment: 关系状态投影片段。
        tension_queue_segment: 张力队列投影片段。
    """

    input_text: str
    dominant_channels: tuple[str, ...] = ()
    has_knots: bool = False
    recalled_surfaces: tuple[str, ...] = ()
    recent_summaries: tuple[str, ...] = ()
    relational_state_segment: str = ""
    tension_queue_segment: str = ""
