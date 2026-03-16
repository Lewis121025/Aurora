"""运行时合约模块。

定义 Aurora 运行时的核心类型和协议：
- Speaker: 说话者枚举
- AuroraMove: Aurora 行为类型
- Turn: 转换记录
- clamp: 数值钳制工具函数
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class Speaker(str, Enum):
    """说话者枚举。

    区分转换记录的来源。
    """

    USER = "user"
    AURORA = "aurora"


AuroraMove = Literal["approach", "withhold", "boundary", "repair", "silence", "witness"]
"""Aurora 行为类型。

- approach: 接近
- withhold: 保留
- boundary: 边界
- repair: 修复
- silence: 沉默
- witness: 见证
"""


@dataclass(frozen=True, slots=True)
class Turn:
    """转换记录。

    记录单次交互的转换信息，用于构建对话历史。

    Attributes:
        turn_id: 转换唯一 ID。
        relation_id: 所属关系 ID。
        session_id: 会话 ID。
        speaker: 说话者。
        text: 文本内容。
        created_at: 创建时间戳。
    """

    turn_id: str
    relation_id: str
    session_id: str
    speaker: Speaker
    text: str
    created_at: float


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """钳制浮点数到指定范围。

    Args:
        value: 待钳制的值。
        lo: 下限，默认 0.0。
        hi: 上限，默认 1.0。

    Returns:
        钳制后的值。
    """
    return max(lo, min(hi, value))
