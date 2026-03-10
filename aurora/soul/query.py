"""
aurora/soul/query.py
查询分析模块：负责解析用户的自然语言查询，识别查询意图（Intent）和时间范围。
它定义了查询类型 (QueryType) 及其关联的启发式关键词库。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

# 意图识别关键词库
TEMPORAL_KEYWORDS = {
    "什么时候", "之前", "之后", "上次", "最近", "以前", "后来", "第一次", "最后",
    "多久", "当时", "那时", "几点", "几月", "几号", "哪天", "哪年", "历史",
    "when", "before", "after", "first", "recently", "earlier", "later",
    "next", "yesterday", "today", "ago", "since", "until", "during",
    "history", "timeline", "chronological",
}

CAUSAL_KEYWORDS = {
    "为什么", "原因", "因为", "所以", "导致", "结果", "因此", "由于", "怎么会",
    "为何", "何故", "缘由", "起因", "影响", "后果",
    "why", "because", "cause", "reason", "result", "therefore", "hence",
    "consequently", "due to", "leads to", "effect", "impact", "outcome",
}

MULTI_HOP_KEYWORDS = {
    "相关", "关联", "联系", "连接", "对比", "比较", "类似", "相似", "区别",
    "所有", "全部", "总结", "概括", "归纳", "涉及", "包含", "关系",
    "related", "connection", "link", "compare", "contrast", "similar", "difference",
    "all", "every", "summarize", "overview", "involve", "contain", "relationship",
    "between", "across", "through",
}

AGGREGATION_KEYWORDS = {
    "多少", "几个", "总数", "总共", "合计", "一共", "累计", "汇总", "总计",
    "所有", "全部", "都", "每", "各",
    "how many", "how much", "total", "sum", "count", "all", "every", "each",
    "aggregate", "combined", "together", "altogether", "in total", "in all",
    "number of", "amount of", "quantity of",
}

# 检索超参数
MULTI_HOP_EXTRA_PAGERANK_ITER = 20

RECENT_ANCHOR_KEYWORDS = {
    "最近", "上次", "刚才", "刚刚", "近期", "这段时间", "最新", "最后",
    "recently", "last time", "just now", "lately", "latest", "most recent",
    "newest", "current",
    "last thing", "last topic", "last event", "last item", "the last",
    "just talked", "just mentioned", "just discussed", "just said",
}

EARLIEST_ANCHOR_KEYWORDS = {
    "最早", "一开始", "起初", "最初", "开始时", "第一次", "首次", "当初", "最先",
    "first", "originally", "initially", "earliest", "beginning", "started",
    "first time", "at first", "in the beginning", "original",
}

SPAN_ANCHOR_KEYWORDS = {
    "一直", "从...到", "之前...之后", "历史", "全部", "所有时候", "整个过程",
    "历程", "演变", "发展过程", "时间线", "变化",
    "throughout", "over time", "history", "timeline", "evolution",
    "all along", "from start", "progression", "across time", "journey",
    "before and after", "development", "changes over",
}

# 评分加成系数
FACTUAL_PLOT_PRIORITY_BOOST = 0.15
FACTUAL_SEMANTIC_WEIGHT = 0.90
FACTUAL_ATTRACTOR_WEIGHT = 0.25
SINGLE_SESSION_USER_K_MULTIPLIER = 2.0
SINGLE_SESSION_USER_MAX_CONTEXT = 15000
KEYWORD_MATCH_BOOST = 0.20
KEYWORD_MATCH_MIN_RATIO = 0.25
USER_ROLE_PRIORITY_BOOST = 0.15
FACT_KEY_BOOST_MAX = 0.15

# 停用词库
QUESTION_STOP_WORDS = {
    "what", "where", "when", "how", "why", "who", "which", "whose", "whom",
    "is", "are", "was", "were", "did", "do", "does", "done", "been", "being",
    "the", "a", "an", "of", "to", "in", "for", "on", "with", "at", "by", "from",
    "my", "your", "i", "you", "me", "we", "they", "it", "our", "their", "its",
    "have", "has", "had", "can", "could", "would", "should", "will", "shall",
    "about", "that", "this", "these", "those", "there", "here",
    "什么", "哪里", "哪个", "谁", "怎么", "为什么", "是", "的", "了", "吗", "我", "你",
}


# 聚合实体提取模式
AGGREGATION_ENTITY_PATTERNS = {
    "camping": ["camping", "camp", "tent", "campsite", "campground"],
    "trip": ["trip", "travel", "visit", "vacation", "journey", "tour"],
    "bike": ["bike", "bicycle", "cycling", "biking", "cycle", "cyclist"],
    "game": ["game", "gaming", "play", "playing", "video game"],
    "book": ["book", "books", "reading", "read", "novel", "library"],
    "movie": ["movie", "film", "watch", "cinema", "theater", "theatre"],
    "exercise": ["exercise", "workout", "gym", "fitness", "training", "sport"],
    "meeting": ["meeting", "call", "appointment", "conference"],
    "doctor": ["doctor", "appointment", "medical", "health", "hospital", "clinic"],
    "art": ["art", "gallery", "museum", "exhibition", "exhibit", "painting", "sculpture"],
    "event": ["event", "concert", "show", "performance", "festival"],
    "model": ["model", "kit", "hobby", "craft", "build", "assemble"],
    "clothing": ["clothing", "clothes", "shirt", "pants", "dress", "jacket", "outfit"],
    "food": ["food", "meal", "restaurant", "dinner", "lunch", "breakfast", "eat"],
    "work": ["work", "job", "project", "task", "assignment"],
    "money": ["money", "spent", "cost", "price", "paid", "bought", "purchase", "$", "dollar", "dollars"],
    "luxury": ["luxury", "expensive", "premium", "high-end"],
    "expense": ["expense", "expenses", "spent", "spending", "cost", "costs"],
    "hour": ["hour", "hours"],
    "day": ["day", "days"],
    "week": ["week", "weeks"],
    "month": ["month", "months"],
    "year": ["year", "years"],
    "total": ["total", "all", "altogether", "sum", "combined"],
    "different": ["different", "various", "unique", "distinct"],
}

AGGREGATION_STOP_WORDS = {
    "what", "where", "when", "how", "why", "who", "which", "whom",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "doing", "the", "a", "an", "this", "that",
    "i", "me", "my", "we", "us", "our", "you", "your",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "and", "but", "or", "so", "many", "much", "some", "any",
    "can", "could", "will", "would", "total", "number", "count",
}

# 身份与用户事实识别库
IDENTITY_KEYWORDS = {
    "你是谁", "你是怎样", "你的性格", "你的价值观", "你的原生人格", "你的前史",
    "who are you", "what are you", "your personality", "your values",
    "your identity", "your backstory", "self narrative", "self-model",
}

USER_FACT_KEYWORDS = {
    "我", "我的", "我喜欢", "我偏好", "我之前",
    "remember my", "about me", "my preference", "my habit", "my profile",
}


class QueryType(Enum):
    """查询类型枚举"""
    FACTUAL = auto()    # 事实检索
    TEMPORAL = auto()   # 时间检索
    MULTI_HOP = auto()  # 多跳联想
    CAUSAL = auto()     # 因果分析
    USER_FACT = auto()  # 用户事实
    IDENTITY = auto()   # 身份认知


class TimeAnchor(Enum):
    """时间锚点类型"""
    RECENT = auto()     # 最近
    EARLIEST = auto()   # 最早
    SPAN = auto()       # 时间跨度
    NONE = auto()


@dataclass(frozen=True)
class TimeRange:
    """时间范围描述对象"""
    start: Optional[float] = None
    end: Optional[float] = None
    anchor_event: Optional[str] = None
    relation: str = "any"

    def to_state_dict(self) -> dict[str, object]:
        """序列化"""
        return {
            "start": self.start,
            "end": self.end,
            "anchor_event": self.anchor_event,
            "relation": self.relation,
        }


def _contains_any(query_lower: str, keywords: set[str]) -> bool:
    """辅助函数：检查文本是否包含关键词集合中的任意项"""
    return any(keyword in query_lower for keyword in keywords)


class QueryAnalyzer:
    """
    查询分析器：分析查询文本并提取结构化意图。
    """
    def classify(self, query_text: str) -> QueryType:
        """分类查询意图"""
        query_lower = query_text.lower()
        temporal_keywords = TEMPORAL_KEYWORDS | EARLIEST_ANCHOR_KEYWORDS | RECENT_ANCHOR_KEYWORDS | SPAN_ANCHOR_KEYWORDS

        if _contains_any(query_lower, temporal_keywords):
            return QueryType.TEMPORAL
        if _contains_any(query_lower, IDENTITY_KEYWORDS):
            return QueryType.IDENTITY
        if _contains_any(query_lower, CAUSAL_KEYWORDS):
            return QueryType.CAUSAL
        if _contains_any(query_lower, USER_FACT_KEYWORDS):
            return QueryType.USER_FACT
        if _contains_any(query_lower, AGGREGATION_KEYWORDS) or _contains_any(query_lower, MULTI_HOP_KEYWORDS):
            return QueryType.MULTI_HOP
        return QueryType.FACTUAL

    def is_aggregation_query(self, query_text: str) -> bool:
        """是否为聚合/统计类查询"""
        return _contains_any(query_text.lower(), AGGREGATION_KEYWORDS)

    def extract_aggregation_entities(self, query_text: str) -> List[str]:
        """为聚合查询提取潜在的实体对象"""
        query_lower = query_text.lower()
        entities: List[str] = []

        # 匹配预定义模式
        for keywords in AGGREGATION_ENTITY_PATTERNS.values():
            if any(keyword in query_lower for keyword in keywords):
                entities.extend(keywords)

        # 简单的正则分词补全
        for word in re.findall(r"\b[a-z]+(?:-[a-z]+)?\b", query_lower):
            if len(word) <= 2 or word in AGGREGATION_STOP_WORDS or word in entities:
                continue
            if len(word) >= 4:
                entities.append(word)

        # 去重
        seen: set[str] = set()
        unique_entities: List[str] = []
        for entity in entities:
            if entity in seen:
                continue
            seen.add(entity)
            unique_entities.append(entity)
        return unique_entities

    def extract_query_keywords(self, query_text: str) -> List[str]:
        """提取查询的核心关键词（去除停用词）"""
        keywords: List[str] = []
        for word in query_text.lower().split():
            clean_word = word.strip("?.,!'\"()[]{}:;")
            if len(clean_word) > 2 and clean_word not in QUESTION_STOP_WORDS:
                keywords.append(clean_word)
        return keywords

    def detect_time_anchor(self, query_text: str) -> TimeAnchor:
        """检测时间锚点类型"""
        query_lower = query_text.lower()
        if _contains_any(query_lower, RECENT_ANCHOR_KEYWORDS):
            return TimeAnchor.RECENT
        if _contains_any(query_lower, EARLIEST_ANCHOR_KEYWORDS):
            return TimeAnchor.EARLIEST
        if _contains_any(query_lower, SPAN_ANCHOR_KEYWORDS):
            return TimeAnchor.SPAN
        return TimeAnchor.NONE


class TimeRangeExtractor:
    """
    时间范围提取器：将自然语言的时间描述（如“上周”）映射为绝对时间戳范围。
    """
    ANCHOR_PATTERNS = {
        "first": list(EARLIEST_ANCHOR_KEYWORDS),
        "last": list(RECENT_ANCHOR_KEYWORDS),
        "span": list(SPAN_ANCHOR_KEYWORDS),
    }
    RELATIVE_PATTERNS = {
        "last_week": r"(last week|上周|上星期)",
        "last_month": r"(last month|上个月|上月)",
        "yesterday": r"(yesterday|昨天)",
        "today": r"(today|今天)",
    }

    def extract(self, query: str, events_timeline: Optional[List[Tuple[str, float]]] = None) -> TimeRange:
        """解析查询并返回时间范围"""
        query_lower = query.lower()
        events_timeline = events_timeline or []

        # 检查显式锚点（最早/最近）
        for relation, keywords in self.ANCHOR_PATTERNS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return self._resolve_anchor(relation, events_timeline)

        # 检查相对时间（上周等）
        for pattern_name, pattern in self.RELATIVE_PATTERNS.items():
            if re.search(pattern, query_lower, re.IGNORECASE):
                return self._resolve_relative_time(pattern_name, events_timeline)

        return TimeRange(relation="any")

    def _resolve_anchor(self, relation: str, events_timeline: List[Tuple[str, float]]) -> TimeRange:
        """将抽象锚点转换为基于现有时间线的范围"""
        if not events_timeline:
            return TimeRange(relation=relation)

        timestamps = [ts for _, ts in events_timeline]
        if relation == "first":
            earliest = min(timestamps)
            return TimeRange(end=earliest + 86400, relation="first") # 第一天
        if relation == "last":
            latest = max(timestamps)
            return TimeRange(start=latest - 86400, relation="last")  # 最后一天
        if relation == "span":
            return TimeRange(relation="span")
        return TimeRange(relation=relation)

    def _resolve_relative_time(self, pattern_name: str, events_timeline: List[Tuple[str, float]]) -> TimeRange:
        """解析相对时间词"""
        if not events_timeline:
            return TimeRange(relation="any")

        latest_ts = max(ts for _, ts in events_timeline)
        # 以记忆中最晚的时间点作为基准“现在”
        if pattern_name == "yesterday":
            return TimeRange(start=latest_ts - 2 * 86400, end=latest_ts - 86400, relation="during")
        if pattern_name == "today":
            return TimeRange(start=latest_ts - 86400, relation="during")
        if pattern_name == "last_week":
            return TimeRange(start=latest_ts - 14 * 86400, end=latest_ts - 7 * 86400, relation="during")
        if pattern_name == "last_month":
            return TimeRange(start=latest_ts - 60 * 86400, end=latest_ts - 30 * 86400, relation="during")
        return TimeRange(relation="any")

    def filter_by_range(
        self,
        candidates: List[Tuple[str, float, float]],
        time_range: TimeRange,
    ) -> List[Tuple[str, float, float]]:
        """应用时间范围过滤器过滤候选项"""
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
