"""Aurora v2 运行时合约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


LoopType = Literal["commitment", "contradiction", "unfinished_thread", "unresolved_question"]
LoopStatus = Literal["active", "resolved"]
FactStatus = Literal["active", "superseded", "disputed"]
MemoryOpType = Literal[
    "assert_fact",
    "revise_fact",
    "patch_relation",
    "open_loop",
    "resolve_loop",
    "add_rule",
    "update_lexicon",
]
EventKind = Literal["user_turn", "assistant_turn", "compile_failure"]
RecallKind = Literal["fact", "event"]


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """钳制浮点数到指定范围。"""
    return max(lo, min(hi, value))


@dataclass(slots=True)
class RelationField:
    """长期主观关系状态。"""

    relation_id: str
    trust: float = 0.35
    distance: float = 0.65
    warmth: float = 0.30
    tension: float = 0.0
    repair_debt: float = 0.0
    shared_lexicon: list[str] = field(default_factory=list)
    interaction_rules: list[str] = field(default_factory=list)
    last_compiled_at: float = 0.0


@dataclass(frozen=True, slots=True)
class OpenLoop:
    """未完成事项或冲突张力。"""

    loop_id: str
    relation_id: str
    loop_type: LoopType
    status: LoopStatus
    summary: str
    urgency: float
    opened_at: float
    updated_at: float
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FactRecord:
    """版本化事实记录。"""

    fact_id: str
    relation_id: str
    content: str
    document_date: float
    event_date: float
    status: FactStatus
    supersedes: str | None
    confidence: float
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EventRecord:
    """Evidence log 事件。"""

    event_id: str
    relation_id: str
    session_id: str
    kind: EventKind
    role: str
    text: str
    created_at: float
    pending_compile: bool
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MemoryOp:
    """Compiler 输出的唯一更新操作。"""

    op_type: MemoryOpType
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RecallHit:
    """检索命中结果。"""

    item_id: str
    kind: RecallKind
    content: str
    score: float
    why_recalled: str
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RecallResult:
    """冷记忆召回结果。"""

    relation_id: str
    query: str
    hits: tuple[RecallHit, ...]


@dataclass(frozen=True, slots=True)
class TurnOutput:
    """热路径 turn 输出。"""

    turn_id: str
    relation_id: str
    response_text: str
    recall_used: bool
    recalled_ids: tuple[str, ...]
    pending_event_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CompileFailure:
    """编译失败记录。"""

    relation_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class CompileReport:
    """后台 compiler 执行结果。"""

    compiled_relations: tuple[str, ...]
    applied_ops: int
    failures: tuple[CompileFailure, ...]


@dataclass(frozen=True, slots=True)
class RelationSnapshot:
    """可调试的关系快照。"""

    relation_id: str
    field: RelationField
    open_loops: tuple[OpenLoop, ...]
    facts: tuple[FactRecord, ...]
    pending_compile_count: int
