"""
aurora/runtime/results.py
运行时结果模型：定义了摄入、查询以及单轮对话交互产生的各种数据结构。
这些类主要用于在 runtime 层、API 层和前端之间传递状态信息。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from aurora.soul.models import IdentitySnapshot, NarrativeSummary


@dataclass
class IngestResult:
    """记忆摄入结果：记录了情节存入后的心理指标和能量变动。"""
    event_id: str
    plot_id: str
    story_id: Optional[str]
    mode: str
    source: Literal["wake", "dream", "repair", "mode"]
    tension: float                 # 该片段产生的叙事张力
    contradiction: float           # 认知失调得分
    active_energy: float           # 摄入后的活动能量
    repressed_energy: float        # 摄入后的压抑能量


@dataclass(frozen=True)
class QueryHit:
    """单个查询命中的描述。"""
    id: str
    kind: str
    score: float
    snippet: str
    metadata: Optional[Dict[str, str]] = None


@dataclass
class QueryResult:
    """完整的查询结果。"""
    query: str
    attractor_path_len: int        # 吸引子追踪的步数
    hits: List[QueryHit]


@dataclass(frozen=True)
class EvidenceRef:
    """证据引用：记录检索命中的具体节点及其在当前交互中的角色。"""
    id: str
    kind: str
    score: float
    role: str                      # 角色，如 "retrieved", "assigned"


@dataclass(frozen=True)
class RetrievalTraceSummary:
    """检索追踪摘要：对一次场论检索过程的统计概括。"""
    query: str
    attractor_path_len: int
    hit_count: int
    ranked_kinds: List[str] = field(default_factory=list)
    query_type: Optional[str] = None
    time_relation: Optional[str] = None
    time_start: Optional[float] = None
    time_end: Optional[float] = None
    time_anchor_event: Optional[str] = None


@dataclass(frozen=True)
class StructuredMemoryContext:
    """结构化记忆上下文：封装了传递给 LLM 之前的完整心理与记忆状态。"""
    mode: str
    narrative_pressure: float      # 当前叙事压力
    intuition: List[str] = field(default_factory=list) # 潜意识直觉关键词
    identity: Optional[IdentitySnapshot] = None        # 身份快照
    narrative_summary: Optional[NarrativeSummary] = None # 自我叙事总结
    retrieval_hits: List[str] = field(default_factory=list) # 检索命中的文本片段
    evidence_refs: List[EvidenceRef] = field(default_factory=list) # 节点引用


@dataclass(frozen=True)
class ChatTimings:
    """对话耗时统计（单位：毫秒）。"""
    retrieval_ms: float
    generation_ms: float
    ingest_ms: float
    total_ms: float


@dataclass(frozen=True)
class ChatTurnResult:
    """单轮对话的完整产出。"""
    reply: str                     # LLM 的回复内容
    event_id: str                  # 本次交互的唯一事件 ID
    memory_context: StructuredMemoryContext # 记忆上下文
    rendered_memory_brief: str     # 渲染后的记忆摘要文本
    system_prompt: str             # 发送给 LLM 的系统提示词
    user_prompt: str               # 发送给 LLM 的用户提示词
    retrieval_trace_summary: RetrievalTraceSummary # 检索统计
    ingest_result: IngestResult    # 摄入结果
    timings: ChatTimings           # 耗时统计
    llm_error: Optional[str] = None # 如果发生错误，记录错误信息


@dataclass(frozen=True)
class ChatStreamEvent:
    """对话流事件：用于支持流式输出。"""
    kind: Literal["status", "reply_delta", "done"]
    stage: Literal["retrieval", "generation", "ingest", "done"]
    text: str = ""                 # 状态描述或回复片段
    result: Optional[ChatTurnResult] = None # 最终生成的完整结果
