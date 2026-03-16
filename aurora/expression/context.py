"""表达上下文模块。

定义 LLM 认知过程的输入上下文结构，包含：
- 用户输入文本
- 主导通道
- 记忆结状态
- 回忆片段
- 最近交互摘要
- 定向快照
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aurora.runtime.contracts import TraceChannel


@dataclass(frozen=True, slots=True)
class ExpressionContext:
    """LLM 认知表达上下文。

    封装单次认知调用所需的全部上下文信息，
    用于构建 LLM 提示词。

    Attributes:
        input_text: 用户输入文本。
        dominant_channels: 主导轨迹通道列表。
        has_knots: 是否存在未解决的记忆结。
        recalled_surfaces: 回忆片段表面描述列表。
        recent_summaries: 最近交互摘要列表。
        orientation_snapshot: 本体定向快照（自我/世界/关系证据）。
    """

    input_text: str
    """用户输入文本。"""

    dominant_channels: tuple[TraceChannel, ...]
    """当前主导轨迹通道列表。"""

    has_knots: bool
    """是否存在未解决的记忆结（张力）。"""

    recalled_surfaces: tuple[str, ...] = ()
    """从记忆检索出的片段表面描述。"""

    recent_summaries: tuple[str, ...] = ()
    """最近交互的摘要列表。"""

    orientation_snapshot: dict[str, Any] | None = None
    """本体定向快照，包含自我/世界/关系三个维度的证据统计。"""

    relation_hint: str = ""
    """关系偏置提示，由 relation/decision 派生，注入认知上下文。"""

    structural_hint: str = ""
    """结构驱动提示，由 expression/structure 派生，包含关系阶段、触碰检测、战略姿态。"""
