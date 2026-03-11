"""
aurora/soul/query.py
查询分析模块：使用 LLM 结构化路由作为唯一主链路，
为检索层提供查询意图、时间规划和聚合线索。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from aurora.integrations.llm.provider import LLMProvider
from aurora.soul.models import Message, TextPart
from aurora.system.errors import ConfigurationError

# 检索超参数
MULTI_HOP_EXTRA_PAGERANK_ITER = 20

FACTUAL_PLOT_PRIORITY_BOOST = 0.15
FACTUAL_SEMANTIC_WEIGHT = 0.90
FACTUAL_ATTRACTOR_WEIGHT = 0.25
SINGLE_SESSION_USER_K_MULTIPLIER = 2.0
SINGLE_SESSION_USER_MAX_CONTEXT = 15000
KEYWORD_MATCH_BOOST = 0.20
KEYWORD_MATCH_MIN_RATIO = 0.25
USER_ROLE_PRIORITY_BOOST = 0.15
FACT_KEY_BOOST_MAX = 0.15


class QueryType(Enum):
    """查询类型枚举"""

    FACTUAL = auto()
    TEMPORAL = auto()
    MULTI_HOP = auto()
    CAUSAL = auto()
    USER_FACT = auto()
    IDENTITY = auto()


TemporalRelation = Literal["any", "first", "last", "span", "during"]
RelativeWindow = Literal["none", "today", "yesterday", "last_week", "last_month"]


@dataclass(frozen=True)
class TemporalPlan:
    """模型产出的时间规划。"""

    relation: TemporalRelation = "any"
    relative_window: RelativeWindow = "none"


@dataclass(frozen=True)
class TimeRange:
    """时间范围描述对象"""

    start: Optional[float] = None
    end: Optional[float] = None
    anchor_event: Optional[str] = None
    relation: str = "any"

    def to_state_dict(self) -> dict[str, object]:
        return {
            "start": self.start,
            "end": self.end,
            "anchor_event": self.anchor_event,
            "relation": self.relation,
        }


@dataclass(frozen=True)
class QueryAnalysis:
    """结构化查询分析结果。"""

    query_type: QueryType
    temporal_plan: TemporalPlan = field(default_factory=TemporalPlan)
    is_aggregation: bool = False
    aggregation_entities: List[str] = field(default_factory=list)
    query_keywords: List[str] = field(default_factory=list)
    query_type_score: float = 0.0
    temporal_score: float = 0.0


class TemporalPlanPayload(BaseModel):
    """LLM 产出的结构化时间规划。"""

    relation: TemporalRelation = "any"
    relative_window: RelativeWindow = "none"


class QueryRoutePayload(BaseModel):
    """LLM query router 的结构化输出。"""

    query_type: Literal["FACTUAL", "TEMPORAL", "MULTI_HOP", "CAUSAL", "USER_FACT", "IDENTITY"]
    temporal_plan: TemporalPlanPayload = Field(default_factory=TemporalPlanPayload)
    is_aggregation: bool = False
    aggregation_entities: List[str] = Field(default_factory=list)
    query_keywords: List[str] = Field(default_factory=list)
    query_type_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    temporal_confidence: float = Field(default=0.7, ge=0.0, le=1.0)


QUERY_ROUTER_SYSTEM_PROMPT = """You are Aurora's only query router.

Classify the user's query for memory retrieval.

Definitions:
- FACTUAL: asks what happened, what something is, or concrete details.
- TEMPORAL: asks about time, sequence, before/after, earliest/latest, or timeline order.
- MULTI_HOP: asks to connect, compare, summarize, or combine multiple memories.
- CAUSAL: asks why something happened, what led to it, or consequences.
- USER_FACT: asks about the user's own profile, habits, preferences, personal history, or stated facts.
- IDENTITY: asks about Aurora's own identity, personality, values, or self-model.

Important disambiguation rules:
- If time words are only the topic, do NOT mark TEMPORAL.
  Example: "那部关于时间线的电影" is FACTUAL unless the user asks about order/time.
- Only mark USER_FACT when the user is asking about themselves, not just using first-person wording in a normal factual question.
- Set is_aggregation=true only when the query explicitly requires counting, listing all, summing, or combining multiple items.
- temporal_plan.relation:
  - any: no temporal filtering or ordering
  - first: earliest / first / beginning
  - last: latest / most recent / last
  - span: full history / over time / before and after / whole timeline
  - during: bounded relative slice such as today / yesterday / last week / last month
- temporal_plan.relative_window:
  - none: not needed
  - today / yesterday / last_week / last_month: only for relation="during"
- If query_type is not TEMPORAL, temporal_plan must be {"relation":"any","relative_window":"none"}.

Return only valid JSON matching the schema."""


def build_query_router_user_prompt(query_text: str) -> str:
    return (
        "Analyze the following user query for Aurora memory retrieval.\n\n"
        f"Query:\n{query_text}\n\n"
        "Return the structured routing result."
    )


QUERY_TYPE_BY_NAME = {
    "FACTUAL": QueryType.FACTUAL,
    "TEMPORAL": QueryType.TEMPORAL,
    "MULTI_HOP": QueryType.MULTI_HOP,
    "CAUSAL": QueryType.CAUSAL,
    "USER_FACT": QueryType.USER_FACT,
    "IDENTITY": QueryType.IDENTITY,
}


def _dedupe_nonempty(items: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    cleaned: List[str] = []
    for item in items:
        text = item.strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


class BaseQueryAnalyzer(ABC):
    """查询分析接口。"""

    def classify(self, query_text: str) -> QueryType:
        return self.analyze(query_text).query_type

    def extract_temporal_plan(self, query_text: str) -> TemporalPlan:
        return self.analyze(query_text).temporal_plan

    def is_aggregation_query(self, query_text: str) -> bool:
        return self.analyze(query_text).is_aggregation

    def extract_aggregation_entities(self, query_text: str) -> List[str]:
        return self.analyze(query_text).aggregation_entities

    def extract_query_keywords(self, query_text: str) -> List[str]:
        return self.analyze(query_text).query_keywords

    @abstractmethod
    def analyze(self, query_text: str) -> QueryAnalysis:
        raise NotImplementedError


class MissingQueryAnalyzer(BaseQueryAnalyzer):
    """显式表示 query analyzer 未配置。"""

    def analyze(self, query_text: str) -> QueryAnalysis:
        raise ConfigurationError(
            "Aurora query routing requires a live LLM provider. Configure llm_provider or inject a query_analyzer."
        )


class LLMQueryAnalyzer(BaseQueryAnalyzer):
    """基于 LLM 的结构化查询路由器。"""

    def __init__(
        self,
        *,
        llm: LLMProvider,
        timeout_s: float = 5.0,
        max_retries: int = 1,
    ) -> None:
        self._llm = llm
        self._timeout_s = timeout_s
        self._max_retries = max_retries

    def analyze(self, query_text: str) -> QueryAnalysis:
        text = query_text.strip()
        if not text:
            return QueryAnalysis(query_type=QueryType.FACTUAL)

        payload = self._llm.complete_json(
            messages=(
                Message(role="system", parts=(TextPart(text=QUERY_ROUTER_SYSTEM_PROMPT),)),
                Message(
                    role="user",
                    parts=(TextPart(text=build_query_router_user_prompt(text)),),
                ),
            ),
            schema=QueryRoutePayload,
            temperature=0.0,
            timeout_s=self._timeout_s,
            metadata={"operation": "query_routing_v5"},
            max_retries=self._max_retries,
        )
        query_type = QUERY_TYPE_BY_NAME[payload.query_type]
        temporal_plan = TemporalPlan(
            relation=payload.temporal_plan.relation,
            relative_window=payload.temporal_plan.relative_window,
        )
        if query_type != QueryType.TEMPORAL:
            temporal_plan = TemporalPlan()
        return QueryAnalysis(
            query_type=query_type,
            temporal_plan=temporal_plan,
            is_aggregation=bool(payload.is_aggregation or payload.aggregation_entities),
            aggregation_entities=_dedupe_nonempty(payload.aggregation_entities),
            query_keywords=_dedupe_nonempty(payload.query_keywords),
            query_type_score=float(payload.query_type_confidence),
            temporal_score=float(payload.temporal_confidence),
        )


class QueryAnalyzer(LLMQueryAnalyzer):
    """Aurora 默认查询分析器。"""


class TimeRangeExtractor:
    """将模型产出的时间规划解析为绝对时间范围。"""

    def extract(
        self,
        events_timeline: Optional[List[Tuple[str, float]]] = None,
        *,
        temporal_plan: Optional[TemporalPlan] = None,
        reference_ts: Optional[float] = None,
    ) -> TimeRange:
        plan = temporal_plan or TemporalPlan()
        events_timeline = events_timeline or []
        relation = plan.relation
        if relation == "any":
            return TimeRange(relation="any")
        if relation in {"first", "last", "span"}:
            return self._resolve_ordered_relation(relation=relation, events_timeline=events_timeline)
        if relation == "during":
            return self._resolve_relative_window(
                relative_window=plan.relative_window,
                events_timeline=events_timeline,
                reference_ts=reference_ts,
            )
        return TimeRange(relation="any")

    def _resolve_ordered_relation(
        self,
        *,
        relation: TemporalRelation,
        events_timeline: List[Tuple[str, float]],
    ) -> TimeRange:
        if not events_timeline:
            if relation == "last":
                return TimeRange(relation="last")
            if relation == "first":
                return TimeRange(relation="first")
            if relation == "span":
                return TimeRange(relation="span")
            return TimeRange(relation="any")

        timestamps = [ts for _, ts in events_timeline]
        if relation == "first":
            earliest = min(timestamps)
            return TimeRange(end=earliest + 86400, relation="first")
        if relation == "last":
            latest = max(timestamps)
            return TimeRange(start=latest - 86400, relation="last")
        if relation == "span":
            return TimeRange(relation="span")
        return TimeRange(relation="any")

    def _resolve_relative_window(
        self,
        *,
        relative_window: RelativeWindow,
        events_timeline: List[Tuple[str, float]],
        reference_ts: Optional[float],
    ) -> TimeRange:
        anchor_ts = self._reference_ts(events_timeline=events_timeline, reference_ts=reference_ts)
        if relative_window == "yesterday":
            return TimeRange(start=anchor_ts - 2 * 86400, end=anchor_ts - 86400, relation="during")
        if relative_window == "today":
            return TimeRange(start=anchor_ts - 86400, relation="during")
        if relative_window == "last_week":
            return TimeRange(
                start=anchor_ts - 14 * 86400,
                end=anchor_ts - 7 * 86400,
                relation="during",
            )
        if relative_window == "last_month":
            return TimeRange(
                start=anchor_ts - 60 * 86400,
                end=anchor_ts - 30 * 86400,
                relation="during",
            )
        return TimeRange(relation="any")

    def _reference_ts(
        self,
        *,
        events_timeline: List[Tuple[str, float]],
        reference_ts: Optional[float],
    ) -> float:
        now_ts = float(reference_ts) if reference_ts is not None else time.time()
        if not events_timeline:
            return now_ts
        latest_event_ts = max(ts for _, ts in events_timeline)
        return max(now_ts, latest_event_ts)

    def filter_by_range(
        self,
        candidates: List[Tuple[str, float, float]],
        time_range: TimeRange,
    ) -> List[Tuple[str, float, float]]:
        if time_range.relation in {"any", "span"}:
            return candidates

        filtered = []
        for candidate_id, score, ts in candidates:
            if time_range.start is not None and ts < time_range.start:
                continue
            if time_range.end is not None and ts > time_range.end:
                continue
            filtered.append((candidate_id, score, ts))

        if time_range.relation == "first":
            filtered.sort(key=lambda item: item[2])
        elif time_range.relation == "last":
            filtered.sort(key=lambda item: item[2], reverse=True)

        return filtered
