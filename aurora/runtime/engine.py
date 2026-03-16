"""Aurora 引擎模块。

实现 Aurora 核心引擎（AuroraEngine），提供：
- 初始化：从持久化加载状态或创建新实例
- handle_turn: 处理用户输入
- doze: 进入 doze 状态
- sleep: 进入 sleep 状态
- health_summary/state_summary: 状态查询
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from aurora.being.metabolic_state import MetabolicState
from aurora.being.orientation import Orientation
from aurora.llm.config import load_llm_config
from aurora.llm.openai_compat import OpenAICompatProvider
from aurora.llm.provider import LLMProvider
from aurora.memory.store import MemoryStore
from aurora.persistence.store import SQLitePersistence
from aurora.phases.awake import run_awake
from aurora.phases.doze import run_doze
from aurora.phases.sleep import run_sleep
from aurora.relation.store import RelationStore
from aurora.runtime.contracts import AuroraMove, Phase
from aurora.runtime.projections import (
    HealthSummary,
    StateSummary,
    project_health_summary,
    project_state_summary,
)
from aurora.runtime.state import RuntimeState


@dataclass(frozen=True, slots=True)
class EngineOutput:
    """turn 输出。

    Attributes:
        turn_id: 用户转换 ID。
        response_text: Aurora 响应文本。
        aurora_move: Aurora 行为选择。
        dominant_channels: 主导通道列表。
    """

    turn_id: str
    response_text: str
    aurora_move: AuroraMove
    dominant_channels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PhaseOutput:
    """相位输出。

    Attributes:
        phase: 相位类型。
        transition_id: 转换 ID。
    """

    phase: Phase
    transition_id: str


class AuroraEngine:
    """Aurora 核心引擎。

    封装运行时状态、记忆存储、关系存储、持久化和 LLM 提供者，
    提供统一的接口处理交互和相位控制。

    Attributes:
        memory_store: 记忆存储。
        relation_store: 关系存储。
        state: 运行时状态。
        persistence: 持久化存储。
        llm: LLM 提供者。
    """

    __slots__ = ("memory_store", "relation_store", "state", "persistence", "llm")

    def __init__(
        self,
        memory_store: MemoryStore,
        relation_store: RelationStore,
        state: RuntimeState,
        persistence: SQLitePersistence,
        llm: LLMProvider,
    ) -> None:
        """初始化引擎。

        Args:
            memory_store: 记忆存储。
            relation_store: 关系存储。
            state: 运行时状态。
            persistence: 持久化存储。
            llm: LLM 提供者。
        """
        self.memory_store = memory_store
        self.relation_store = relation_store
        self.state = state
        self.persistence = persistence
        self.llm = llm

    @classmethod
    def create(
        cls,
        data_dir: str | None = None,
        llm: LLMProvider | None = None,
    ) -> "AuroraEngine":
        """创建引擎实例。

        从持久化加载状态，若不存在则创建初始状态。
        若未提供 LLM 提供者则从环境变量加载配置。

        Args:
            data_dir: 数据目录，默认 `.aurora`。
            llm: LLM 提供者，默认从环境变量加载。

        Returns:
            AuroraEngine: 引擎实例。

        Raises:
            RuntimeError: LLM 配置缺失。
        """
        persistence = SQLitePersistence(data_dir=data_dir)
        initial = RuntimeState(orientation=Orientation(), metabolic=MetabolicState())
        memory_store, relation_store, state = persistence.load_runtime(initial=initial)

        if llm is None:
            llm_config = load_llm_config()
            if llm_config is None:
                raise RuntimeError(
                    "Aurora requires an LLM provider. "
                    "Set AURORA_LLM_BASE_URL, AURORA_LLM_API_KEY, and AURORA_LLM_MODEL."
                )
            llm = OpenAICompatProvider(llm_config)

        return cls(memory_store, relation_store, state, persistence, llm)

    def handle_turn(self, session_id: str, text: str) -> EngineOutput:
        """处理用户输入。

        执行 awake 相位，创建记忆结构，持久化结果。

        Args:
            session_id: 会话 ID。
            text: 用户输入文本。

        Returns:
            EngineOutput: turn 输出。
        """
        now_ts = time.time()
        relation_id = f"rel:{session_id}"

        outcome = run_awake(
            relation_id=relation_id,
            session_id=session_id,
            text=text,
            orientation=self.state.orientation,
            metabolic=self.state.metabolic,
            memory_store=self.memory_store,
            relation_store=self.relation_store,
            now_ts=now_ts,
            llm=self.llm,
        )

        if outcome.transition is not None:
            self.state.append_transition(outcome.transition)

        self.persistence.persist_awake(
            outcome=outcome,
            state=self.state,
            memory_store=self.memory_store,
            relation_store=self.relation_store,
        )

        return EngineOutput(
            turn_id=outcome.user_turn.turn_id,
            response_text=outcome.response_text,
            aurora_move=outcome.aurora_move,
            dominant_channels=tuple(channel.value for channel in outcome.dominant_channels),
        )

    def doze(self) -> PhaseOutput:
        """进入 doze 状态。

        执行 doze 相位，维护记忆，持久化结果。

        Returns:
            PhaseOutput: 相位输出。
        """
        outcome = run_doze(
            metabolic=self.state.metabolic,
            memory_store=self.memory_store,
            now_ts=time.time(),
        )
        self.state.append_transition(outcome.transition)
        self.persistence.persist_phase(
            outcome=outcome,
            state=self.state,
            memory_store=self.memory_store,
            relation_store=self.relation_store,
        )
        return PhaseOutput(
            phase=outcome.phase,
            transition_id=outcome.transition.transition_id,
        )

    def sleep(self) -> PhaseOutput:
        """进入 sleep 状态。

        执行 sleep 相位，整合记忆，持久化结果。

        Returns:
            PhaseOutput: 相位输出。
        """
        outcome = run_sleep(
            metabolic=self.state.metabolic,
            orientation=self.state.orientation,
            memory_store=self.memory_store,
            relation_store=self.relation_store,
            now_ts=time.time(),
            llm=self.llm,
        )
        self.state.append_transition(outcome.transition)
        self.persistence.persist_phase(
            outcome=outcome,
            state=self.state,
            memory_store=self.memory_store,
            relation_store=self.relation_store,
        )
        return PhaseOutput(
            phase=outcome.phase,
            transition_id=outcome.transition.transition_id,
        )

    def health_summary(self) -> HealthSummary:
        """获取健康摘要。

        Returns:
            HealthSummary: 健康摘要。
        """
        return project_health_summary(
            state=self.state,
            turns=self.persistence.turn_count(),
            transitions=self.persistence.phase_transition_count(),
        )

    def state_summary(self) -> StateSummary:
        """获取状态摘要。

        Returns:
            StateSummary: 状态摘要。
        """
        return project_state_summary(
            state=self.state,
            memory_store=self.memory_store,
            relation_store=self.relation_store,
            turns=self.persistence.turn_count(),
            transitions=self.persistence.phase_transition_count(),
        )
