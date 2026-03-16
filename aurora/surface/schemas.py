"""HTTP API Schema 模块。

定义 FastAPI 使用的请求/响应模型，基于 Pydantic：
- TurnRequest: turn 请求
- TurnResponse: turn 响应
- PhaseResponse: 相位响应
- HealthResponse: 健康检查响应
- StateResponse: 状态查询响应
"""
from __future__ import annotations

from pydantic import BaseModel

from aurora.runtime.contracts import Phase


class TurnRequest(BaseModel):
    """Turn 请求模型。

    Attributes:
        session_id: 会话 ID。
        text: 用户输入文本。
    """

    session_id: str
    text: str


class TurnResponse(BaseModel):
    """Turn 响应模型。

    Attributes:
        turn_id: 转换 ID。
        response_text: Aurora 响应文本。
        aurora_move: Aurora 行为选择。
        dominant_channels: 主导通道列表。
    """

    turn_id: str
    response_text: str
    aurora_move: str
    dominant_channels: tuple[str, ...]


class PhaseResponse(BaseModel):
    """相位响应模型。

    Attributes:
        phase: 相位类型。
        transition_id: 转换 ID。
    """

    phase: Phase
    transition_id: str


class HealthResponse(BaseModel):
    """健康检查响应模型。

    Attributes:
        status: 状态（"ok"）。
        phase: 当前相位。
        turns: 转换次数。
        transitions: 相位转换次数。
    """

    status: str
    phase: str
    turns: int
    transitions: int


class StateResponse(BaseModel):
    """状态查询响应模型。

    Attributes:
        phase: 当前相位。
        sleep_need: 睡眠需求（0.0–1.0）。
        active_relation_ids: 活跃关系 ID 列表。
        pending_sleep_relation_ids: 待处理关系 ID 列表。
        active_knot_ids: 活跃记忆结 ID 列表。
        anchor_thread_ids: 锚定线程 ID 列表。
        turns: 转换次数。
        memory_fragments: 记忆片段数。
        memory_traces: 记忆轨迹数。
        memory_associations: 关联边数。
        memory_threads: 记忆线程数。
        memory_knots: 记忆结数。
        relation_formations: 关系形成记录数。
        relation_moments: 关系时刻数。
        sleep_cycles: sleep 周期数。
        transitions: 相位转换次数。
    """

    phase: str
    sleep_need: float
    active_relation_ids: tuple[str, ...]
    pending_sleep_relation_ids: tuple[str, ...]
    active_knot_ids: tuple[str, ...]
    anchor_thread_ids: tuple[str, ...]
    turns: int
    memory_fragments: int
    memory_traces: int
    memory_associations: int
    memory_threads: int
    memory_knots: int
    relation_formations: int
    relation_moments: int
    sleep_cycles: int
    transitions: int
