"""Aurora 引擎模块。

实现 Aurora 核心引擎（AuroraEngine）：
- handle_turn: 处理用户输入
- 会话轮数累积与蒸馏触发
- RelationalState + TensionQueue 投影到 System Prompt
- 蒸馏产出的原子事实沉淀到 ObjectiveLedger
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from uuid import uuid4

from aurora.llm.config import load_llm_config, DISTILL_THRESHOLD_TURNS, SESSION_IDLE_TIMEOUT_MINUTES
from aurora.llm.openai_compat import OpenAICompatProvider
from aurora.llm.provider import LLMProvider
from aurora.memory.ledger import ObjectiveLedger
from aurora.memory.store import MemoryStore
from aurora.pipelines.distillation import apply_distillation, distill_session
from aurora.relation.state import RelationalState
from aurora.relation.tension import TensionQueue
from aurora.runtime.contracts import AuroraMove
from aurora.runtime.state import RuntimeState
from aurora.phases.awake import run_awake


@dataclass(frozen=True, slots=True)
class EngineOutput:
    """turn 输出。

    Attributes:
        turn_id: 用户转换 ID。
        response_text: Aurora 响应文本。
        aurora_move: Aurora 行为选择。
    """

    turn_id: str
    response_text: str
    aurora_move: AuroraMove


class AuroraEngine:
    """Aurora 核心引擎。

    三层架构：
    - ObjectiveLedger: 冷事实持久化（SQLite + 向量）
    - RelationalState: 主观状态投影（全量挂载到 System Prompt）
    - TensionQueue: 未解决悬案（半衰期衰减）
    """

    __slots__ = (
        "memory_store",
        "ledger",
        "relational_states",
        "tension_queues",
        "state",
        "llm",
        "_conversation_buffer",
    )

    def __init__(
        self,
        memory_store: MemoryStore,
        ledger: ObjectiveLedger,
        relational_states: dict[str, RelationalState],
        tension_queues: dict[str, TensionQueue],
        state: RuntimeState,
        llm: LLMProvider,
    ) -> None:
        self.memory_store = memory_store
        self.ledger = ledger
        self.relational_states = relational_states
        self.tension_queues = tension_queues
        self.state = state
        self.llm = llm
        self._conversation_buffer: dict[str, list[tuple[str, str]]] = {}

    @classmethod
    def create(
        cls,
        data_dir: str | None = None,
        llm: LLMProvider | None = None,
    ) -> "AuroraEngine":
        """创建引擎实例。"""
        if llm is None:
            llm_config = load_llm_config()
            if llm_config is None:
                raise RuntimeError(
                    "Aurora requires an LLM provider. "
                    "Set AURORA_LLM_BASE_URL, AURORA_LLM_API_KEY, and AURORA_LLM_MODEL."
                )
            llm = OpenAICompatProvider(llm_config)

        db_dir = data_dir or ".aurora"
        memory_store = MemoryStore()
        ledger = ObjectiveLedger(db_path=f"{db_dir}/ledger.db")
        relational_states: dict[str, RelationalState] = {}
        tension_queues: dict[str, TensionQueue] = {}
        state = RuntimeState()

        return cls(memory_store, ledger, relational_states, tension_queues, state, llm)

    def handle_turn(
        self, session_id: str, text: str, relation_id: str | None = None
    ) -> EngineOutput:
        """处理用户输入。

        Args:
            session_id: 会话 ID。
            text: 用户输入文本。
            relation_id: 关系 ID（可选，默认使用 session_id）。

        Returns:
            EngineOutput: turn 输出。
        """
        now_ts = time.time()
        rid = relation_id or session_id

        if rid not in self.relational_states:
            self.relational_states[rid] = RelationalState()
        if rid not in self.tension_queues:
            self.tension_queues[rid] = TensionQueue()
        if rid not in self._conversation_buffer:
            self._conversation_buffer[rid] = []

        relational_state = self.relational_states[rid]
        tension_queue = self.tension_queues[rid]

        outcome = run_awake(
            relation_id=rid,
            session_id=session_id,
            text=text,
            relational_state=relational_state,
            tension_queue=tension_queue,
            memory_store=self.memory_store,
            ledger=self.ledger,
            now_ts=now_ts,
            llm=self.llm,
        )

        self._conversation_buffer[rid].append((text, outcome.response_text))

        turn_count = self.state.record_turn(rid, now_ts)

        if turn_count >= DISTILL_THRESHOLD_TURNS:
            self._trigger_distillation(rid, now_ts)
            self.state.session_turn_counts[rid] = 0
            self._conversation_buffer[rid] = []

        return EngineOutput(
            turn_id=outcome.user_turn.turn_id,
            response_text=outcome.response_text,
            aurora_move=outcome.aurora_move,
        )

    def on_session_idle(self, session_id: str | None = None) -> None:
        """会话空闲触发蒸馏。

        Args:
            session_id: 会话 ID（可选，默认处理所有关系）。
        """
        now_ts = time.time()
        timeout_seconds = SESSION_IDLE_TIMEOUT_MINUTES * 60

        if self.state.is_idle(now_ts, timeout_seconds):
            relations = [session_id] if session_id else list(self.relational_states.keys())
            for rid in relations:
                if self._conversation_buffer.get(rid):
                    self._trigger_distillation(rid, now_ts)

    def _trigger_distillation(self, relation_id: str, now_ts: float) -> None:
        """触发蒸馏管道。

        Args:
            relation_id: 关系 ID。
            now_ts: 当前时间戳。
        """
        conversation = self._conversation_buffer.get(relation_id, [])
        if not conversation:
            return

        relational_state = self.relational_states.get(relation_id)
        tension_queue = self.tension_queues.get(relation_id)

        if relational_state is None or tension_queue is None:
            return

        existing_facts = self.ledger.facts_for_relation(relation_id)

        patch = distill_session(
            conversation=conversation,
            relation_id=relation_id,
            current_state=relational_state,
            existing_facts=existing_facts,
            now_ts=now_ts,
            llm=self.llm,
        )

        apply_distillation(
            patch=patch,
            relational_state=relational_state,
            tension_queue=tension_queue,
            now_ts=now_ts,
        )

        if patch.facts:
            for fact_text in patch.facts:
                self.ledger.add_fact(
                    fact_id=f"fact_{uuid4().hex[:12]}",
                    content=fact_text,
                    document_date=now_ts,
                    event_date=now_ts,
                    relation_id=relation_id,
                )

        relational_state.last_distilled_at = now_ts

        self._conversation_buffer[relation_id] = []
