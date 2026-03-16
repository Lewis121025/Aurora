"""awake 相位执行模块。

实现 awake 相位的完整逻辑：
1. 接收用户输入
2. 检索近期记忆
3. 调用 LLM 认知
4. 创建用户和 Aurora 片段
5. 记录轨迹和关联
6. 更新定向和代谢状态
7. 判断是否需要相位转换
"""
from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from aurora.being.metabolic_state import MetabolicState
from aurora.being.orientation import Orientation
from aurora.expression.cognition import run_cognition
from aurora.expression.context import ExpressionContext
from aurora.llm.provider import LLMProvider
from aurora.relation.decision import compute_bias
from aurora.memory.fragment import Fragment
from aurora.memory.recall import build_activation_channels, recent_recall
from aurora.memory.store import MemoryStore
from aurora.memory.trace import Trace
from aurora.phases.transitions import phase_transition
from aurora.relation.moment import RelationMoment
from aurora.relation.store import RelationStore
from aurora.runtime.contracts import (
    AuroraMove,
    Phase,
    PhaseTransition,
    Speaker,
    TraceChannel,
    Turn,
    clamp,
)

SURFACE_LIMIT = 180  # 片段表面文本长度上限

# 用户片段参数
USER_BASE_VIVIDNESS = 0.36
USER_VIVIDNESS_SCALE = 360.0
USER_BASE_SALIENCE = 0.52

# Aurora 片段参数
AURORA_VIVIDNESS = 0.42
AURORA_SALIENCE = 0.44

RECALL_TOUCH_DELTA = 0.03  # 回忆片段触碰增量

# 睡眠需求增量参数
SLEEP_BUMP_BASE = 0.08
SLEEP_BUMP_RECALL_FACTOR = 0.03
SLEEP_BUMP_BOUNDARY_BONUS = 0.04
SLEEP_BUMP_BOUNDARY_MOVE_BONUS = 0.03

SUMMARY_TRUNCATE = 70  # 关系时刻摘要截断长度

# 行为到关联权重的映射
_MOVE_ASSOCIATION_WEIGHT: dict[str, float] = {
    "repair": 0.82,
    "approach": 0.72,
    "witness": 0.68,
    "withhold": 0.52,
    "boundary": 0.44,
    "silence": 0.36,
}

# 行为到强度缩放的映射
_MOVE_INTENSITY_SCALE: dict[str, float] = {
    "approach": 1.0,
    "repair": 1.0,
    "witness": 0.85,
    "withhold": 0.6,
    "boundary": 0.7,
    "silence": 0.45,
}

# 行为到携带度的映射
_MOVE_CARRY_SCALE: dict[str, float] = {
    "approach": 1.1,
    "repair": 1.2,
    "witness": 1.0,
    "withhold": 0.7,
    "boundary": 0.8,
    "silence": 0.5,
}

# 行为到通道增强的映射
_MOVE_CHANNEL_BOOST: dict[str, TraceChannel] = {
    "approach": TraceChannel.WARMTH,
    "boundary": TraceChannel.BOUNDARY,
    "repair": TraceChannel.REPAIR,
    "withhold": TraceChannel.DISTANCE,
    "witness": TraceChannel.RECOGNITION,
}


@dataclass(frozen=True, slots=True)
class AwakeOutcome:
    """awake 相位产出。

    Attributes:
        user_turn: 用户转换记录。
        aurora_turn: Aurora 转换记录。
        user_fragment: 用户片段。
        aurora_fragment: Aurora 片段。
        user_traces: 用户轨迹列表。
        aurora_traces: Aurora 轨迹列表。
        response_text: Aurora 响应文本。
        aurora_move: Aurora 行为选择。
        dominant_channels: 主导通道列表。
        relation_moment: 关系时刻记录。
        transition: 相位转换（若发生）。
    """

    user_turn: Turn
    aurora_turn: Turn
    user_fragment: Fragment
    aurora_fragment: Fragment
    user_traces: tuple[Trace, ...]
    aurora_traces: tuple[Trace, ...]
    response_text: str
    aurora_move: AuroraMove
    dominant_channels: tuple[TraceChannel, ...]
    relation_moment: RelationMoment
    transition: PhaseTransition | None


def run_awake(
    relation_id: str,
    session_id: str,
    text: str,
    orientation: Orientation,
    metabolic: MetabolicState,
    memory_store: MemoryStore,
    relation_store: RelationStore,
    now_ts: float,
    llm: LLMProvider,
) -> AwakeOutcome:
    """执行 awake 相位。

    处理用户输入，执行认知过程，创建记忆结构，更新状态。

    Args:
        relation_id: 关系 ID。
        session_id: 会话 ID。
        text: 用户输入文本。
        orientation: 本体定向。
        metabolic: 代谢状态。
        memory_store: 记忆存储。
        relation_store: 关系存储。
        now_ts: 当前时间戳。
        llm: LLM 提供者。

    Returns:
        AwakeOutcome: awake 相位产出。
    """
    previous_phase = metabolic.phase
    metabolic.enter_phase(Phase.AWAKE, now_ts)
    metabolic.set_active_relation(relation_id)

    # 检索近期记忆
    prior_recalled = recent_recall(
        memory_store, relation_id=relation_id, limit=8, now_ts=now_ts
    )
    prior_channels = build_activation_channels(memory_store, prior_recalled)
    recent_moments = relation_store.moments.get(relation_id, [])

    # 构建认知上下文
    formation = relation_store.formation_for(relation_id)
    bias = compute_bias(formation, now_ts)

    context = ExpressionContext(
        input_text=text,
        dominant_channels=prior_channels,
        has_knots=bool(memory_store.knots_for_relation(relation_id)),
        recalled_surfaces=tuple(f.surface for f in prior_recalled[:4]),
        recent_summaries=tuple(m.summary for m in recent_moments[-3:]),
        orientation_snapshot=orientation.snapshot(),
        relation_hint=bias.hint,
    )

    # 执行认知
    cognition = run_cognition(context, llm)

    # 创建用户转换和片段
    user_turn = Turn(
        turn_id=f"turn_{uuid4().hex[:12]}",
        relation_id=relation_id,
        session_id=session_id,
        speaker=Speaker.USER,
        text=text,
        created_at=now_ts,
    )
    user_fragment = memory_store.create_fragment(
        relation_id=relation_id,
        turn_id=user_turn.turn_id,
        surface=text[:SURFACE_LIMIT],
        vividness=min(1.0, USER_BASE_VIVIDNESS + len(text) / USER_VIVIDNESS_SCALE),
        salience=USER_BASE_SALIENCE,
        unresolvedness=cognition.fragment_unresolvedness,
        now_ts=now_ts,
    )
    user_traces = tuple(
        memory_store.create_trace(
            relation_id=relation_id,
            fragment_id=user_fragment.fragment_id,
            channel=channel,
            intensity=intensity,
            now_ts=now_ts,
        )
        for channel, intensity in cognition.touch_channels
    )

    # 触碰回忆片段
    for fragment in prior_recalled:
        memory_store.touch_fragment(
            fragment.fragment_id, at=now_ts, delta_salience=RECALL_TOUCH_DELTA
        )

    dominant_channels = build_activation_channels(
        memory_store, prior_recalled + (user_fragment,)
    )

    # 创建 Aurora 转换和片段
    aurora_turn = Turn(
        turn_id=f"turn_{uuid4().hex[:12]}",
        relation_id=relation_id,
        session_id=session_id,
        speaker=Speaker.AURORA,
        text=cognition.response_text,
        created_at=now_ts,
    )
    aurora_fragment = memory_store.create_fragment(
        relation_id=relation_id,
        turn_id=aurora_turn.turn_id,
        surface=cognition.response_text[:SURFACE_LIMIT],
        vividness=AURORA_VIVIDNESS,
        salience=AURORA_SALIENCE,
        unresolvedness=cognition.fragment_unresolvedness,
        now_ts=now_ts,
    )

    # 创建 Aurora 轨迹（应用行为缩放）
    move_str = cognition.move
    intensity_scale = _MOVE_INTENSITY_SCALE.get(move_str, 0.85)
    carry_scale = _MOVE_CARRY_SCALE.get(move_str, 1.0)
    boosted_channel = _MOVE_CHANNEL_BOOST.get(move_str)
    aurora_traces = tuple(
        memory_store.create_trace(
            relation_id=relation_id,
            fragment_id=aurora_fragment.fragment_id,
            channel=channel,
            intensity=clamp(intensity * intensity_scale * (1.15 if channel is boosted_channel else 1.0)),
            now_ts=now_ts,
            carry=clamp(0.35 * carry_scale + intensity * 0.5 * carry_scale),
        )
        for channel, intensity in cognition.touch_channels
    )

    # 创建片段关联
    assoc_weight = _MOVE_ASSOCIATION_WEIGHT.get(move_str, 0.68)
    memory_store.link_fragments(
        src_fragment_id=user_fragment.fragment_id,
        dst_fragment_id=aurora_fragment.fragment_id,
        kind=cognition.association_kind,
        weight=assoc_weight,
        evidence=(user_turn.turn_id, aurora_turn.turn_id),
        now_ts=now_ts,
    )

    # 记录关系时刻
    relation_moment = relation_store.record_exchange(
        relation_id=relation_id,
        user_turn_id=user_turn.turn_id,
        aurora_turn_id=aurora_turn.turn_id,
        user_channels=tuple(trace.channel for trace in user_traces),
        aurora_move=cognition.move,
        summary=f"{user_turn.text[:SUMMARY_TRUNCATE]} -> {cognition.response_text[:SUMMARY_TRUNCATE]}",
        now_ts=now_ts,
    )

    # 更新定向
    orientation.register_exchange(
        user_channels=tuple(trace.channel for trace in user_traces),
        aurora_move=cognition.move,
        relation_moment_id=relation_moment.moment_id,
        user_fragment_id=user_fragment.fragment_id,
        aurora_fragment_id=aurora_fragment.fragment_id,
        now_ts=now_ts,
    )

    # 增加睡眠需求
    sleep_bump = SLEEP_BUMP_BASE + SLEEP_BUMP_RECALL_FACTOR * min(len(prior_recalled), 6) / 6.0
    if TraceChannel.BOUNDARY in {trace.channel for trace in user_traces}:
        sleep_bump += SLEEP_BUMP_BOUNDARY_BONUS
    if cognition.move == "boundary":
        sleep_bump += SLEEP_BUMP_BOUNDARY_MOVE_BONUS
    metabolic.bump_sleep_need(sleep_bump)
    metabolic.queue_relation_for_sleep(relation_id)

    # 相位转换
    transition = None
    if previous_phase is not Phase.AWAKE:
        transition = phase_transition(previous_phase, Phase.AWAKE, "incoming_turn", now_ts)

    return AwakeOutcome(
        user_turn=user_turn,
        aurora_turn=aurora_turn,
        user_fragment=user_fragment,
        aurora_fragment=aurora_fragment,
        user_traces=user_traces,
        aurora_traces=aurora_traces,
        response_text=cognition.response_text,
        aurora_move=cognition.move,
        dominant_channels=dominant_channels,
        relation_moment=relation_moment,
        transition=transition,
    )
