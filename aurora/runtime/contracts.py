"""运行时合约模块。

定义 Aurora 运行时的核心类型和协议：
- Phase: 相位枚举（awake/doze/sleep）
- Speaker: 说话者枚举（user/aurora）
- TraceChannel: 轨迹通道枚举
- AssocKind: 关联类型枚举
- AuroraMove: Aurora 行为类型
- Turn: 转换记录
- PhaseTransition: 相位转换记录
- clamp: 数值钳制工具函数
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class Phase(str, Enum):
    """相位枚举。

    定义 Aurora 的三种运行状态：
    - AWAKE: 活跃交互状态
    - DOZE: 低活跃维护状态
    - SLEEP: 记忆整合状态
    """

    AWAKE = "awake"
    DOZE = "doze"
    SLEEP = "sleep"


class Speaker(str, Enum):
    """说话者枚举。

    区分转换记录的来源。
    """

    USER = "user"
    AURORA = "aurora"


class TraceChannel(str, Enum):
    """轨迹通道枚举。

    定义认知过程中可能触发的通道类型：
    - WARMTH: 温暖感
    - HURT: 伤害感
    - RECOGNITION: 被识别感
    - DISTANCE: 距离感
    - CURIOSITY: 好奇心
    - BOUNDARY: 边界感
    - REPAIR: 修复意愿
    - COHERENCE: 连贯性
    - WONDER: 惊奇感
    """

    WARMTH = "warmth"
    HURT = "hurt"
    RECOGNITION = "recognition"
    DISTANCE = "distance"
    CURIOSITY = "curiosity"
    BOUNDARY = "boundary"
    REPAIR = "repair"
    COHERENCE = "coherence"
    WONDER = "wonder"


class AssocKind(str, Enum):
    """关联类型枚举。

    定义记忆片段间关联边的类型：
    - RESONANCE: 共鸣关联
    - CONTRAST: 对比关联
    - REPAIR: 修复关联
    - BOUNDARY: 边界关联
    - THREAD: 线程关联
    - RELATION: 关系关联
    - TEMPORAL: 时间关联
    - KNOT: 记忆结关联
    """

    RESONANCE = "resonance"
    CONTRAST = "contrast"
    REPAIR = "repair"
    BOUNDARY = "boundary"
    THREAD = "thread"
    RELATION = "relation"
    TEMPORAL = "temporal"
    KNOT = "knot"


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


@dataclass(frozen=True, slots=True)
class PhaseTransition:
    """相位转换记录。

    记录相位转换事件，用于状态追踪和持久化。

    Attributes:
        transition_id: 转换唯一 ID。
        from_phase: 源相位。
        to_phase: 目标相位。
        reason: 转换原因。
        created_at: 创建时间戳。
    """

    transition_id: str
    from_phase: Phase
    to_phase: Phase
    reason: str
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
