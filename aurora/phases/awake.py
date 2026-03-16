"""awake 阶段执行模块。

实现极简的对话处理：
1. 接收用户输入
2. 加载关系状态到 System Prompt
3. 调用 LLM 认知
4. 创建记忆节点
5. 累积会话轮数
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from aurora.expression.cognition import run_cognition
from aurora.expression.context import ExpressionContext
from aurora.llm.provider import LLMProvider
from aurora.memory.store import MemoryStore
from aurora.relation.state import RelationalState
from aurora.relation.tension import TensionQueue
from aurora.runtime.contracts import AuroraMove, Speaker, Turn


@dataclass(frozen=True, slots=True)
class AwakeOutcome:
    """awake 阶段产出。

    Attributes:
        user_turn: 用户转换记录。
        aurora_turn: Aurora 转换记录。
        response_text: Aurora 响应文本。
        aurora_move: Aurora 行为选择。
    """

    user_turn: Turn
    aurora_turn: Turn
    response_text: str
    aurora_move: AuroraMove


def run_awake(
    relation_id: str,
    session_id: str,
    text: str,
    relational_state: RelationalState,
    tension_queue: TensionQueue,
    memory_store: MemoryStore,
    now_ts: float,
    llm: LLMProvider,
) -> AwakeOutcome:
    """执行 awake 阶段。

    处理用户输入，执行认知过程，创建记忆节点。

    Args:
        relation_id: 关系 ID。
        session_id: 会话 ID。
        text: 用户输入文本。
        relational_state: 关系状态。
        tension_queue: 张力队列。
        memory_store: 记忆存储。
        now_ts: 当前时间戳。
        llm: LLM 提供者。

    Returns:
        AwakeOutcome: awake 阶段产出。
    """
    user_turn = Turn(
        turn_id=f"turn_{uuid4().hex[:12]}",
        relation_id=relation_id,
        session_id=session_id,
        speaker=Speaker.USER,
        text=text,
        created_at=now_ts,
    )

    prior_nodes = memory_store.nodes_for_relation(relation_id)
    recent_surfaces = tuple(n.content for n in prior_nodes[-4:])

    context = ExpressionContext(
        input_text=text,
        dominant_channels=(),
        has_knots=False,
        recalled_surfaces=recent_surfaces,
        recent_summaries=(),
        relational_state_segment=relational_state.to_prompt_segment(),
        tension_queue_segment=tension_queue.to_prompt_segment(now_ts),
    )

    cognition = run_cognition(context, llm)

    aurora_turn = Turn(
        turn_id=f"turn_{uuid4().hex[:12]}",
        relation_id=relation_id,
        session_id=session_id,
        speaker=Speaker.AURORA,
        text=cognition.response_text,
        created_at=now_ts,
    )

    memory_store.create_node(
        relation_id=relation_id,
        content=text,
        now_ts=now_ts,
    )
    memory_store.create_node(
        relation_id=relation_id,
        content=cognition.response_text,
        now_ts=now_ts,
    )

    return AwakeOutcome(
        user_turn=user_turn,
        aurora_turn=aurora_turn,
        response_text=cognition.response_text,
        aurora_move=cognition.move,
    )
