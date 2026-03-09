"""
AURORA 查询分析
====================

查询类型分类、时间锚点识别和聚合/关键词提取逻辑。
"""

from __future__ import annotations

import re
from enum import Enum, auto
from typing import List

from aurora.core.config.query_types import (
    AGGREGATION_KEYWORDS,
    CAUSAL_KEYWORDS,
    EARLIEST_ANCHOR_KEYWORDS,
    MULTI_HOP_KEYWORDS,
    QUESTION_STOP_WORDS,
    RECENT_ANCHOR_KEYWORDS,
    SPAN_ANCHOR_KEYWORDS,
    TEMPORAL_KEYWORDS,
)


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
    "what",
    "where",
    "when",
    "how",
    "why",
    "who",
    "which",
    "whom",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "i",
    "me",
    "my",
    "mine",
    "we",
    "us",
    "our",
    "ours",
    "you",
    "your",
    "yours",
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
    "it",
    "its",
    "they",
    "them",
    "their",
    "theirs",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "up",
    "about",
    "into",
    "over",
    "after",
    "and",
    "but",
    "or",
    "nor",
    "so",
    "yet",
    "many",
    "much",
    "some",
    "any",
    "few",
    "more",
    "most",
    "can",
    "could",
    "will",
    "would",
    "shall",
    "should",
    "may",
    "might",
    "must",
    "need",
    "dare",
    "used",
    "total",
    "number",
    "amount",
    "count",
    "sum",
}


class QueryType(Enum):
    """用于自适应检索策略的查询类型。"""

    FACTUAL = auto()
    TEMPORAL = auto()
    MULTI_HOP = auto()
    CAUSAL = auto()
    USER_FACT = auto()


class TimeAnchor(Enum):
    """时间查询的时间锚点。"""

    RECENT = auto()
    EARLIEST = auto()
    SPAN = auto()
    NONE = auto()


def _contains_any(query_lower: str, keywords: set[str]) -> bool:
    return any(keyword in query_lower for keyword in keywords)


class QueryAnalysisMixin:
    """查询分类和关键词提取逻辑。"""

    def _classify_query(self, query_text: str) -> QueryType:
        query_lower = query_text.lower()
        temporal_keywords = (
            TEMPORAL_KEYWORDS
            | EARLIEST_ANCHOR_KEYWORDS
            | RECENT_ANCHOR_KEYWORDS
            | SPAN_ANCHOR_KEYWORDS
        )

        if _contains_any(query_lower, temporal_keywords):
            return QueryType.TEMPORAL
        if _contains_any(query_lower, CAUSAL_KEYWORDS):
            return QueryType.CAUSAL
        if _contains_any(query_lower, AGGREGATION_KEYWORDS):
            return QueryType.MULTI_HOP
        if _contains_any(query_lower, MULTI_HOP_KEYWORDS):
            return QueryType.MULTI_HOP
        return QueryType.FACTUAL

    def _is_aggregation_query(self, query_text: str) -> bool:
        return _contains_any(query_text.lower(), AGGREGATION_KEYWORDS)

    def _extract_aggregation_entities(self, query_text: str) -> List[str]:
        query_lower = query_text.lower()
        entities: List[str] = []

        for keywords in AGGREGATION_ENTITY_PATTERNS.values():
            if any(keyword in query_lower for keyword in keywords):
                entities.extend(keywords)

        for word in re.findall(r"\b[a-z]+(?:-[a-z]+)?\b", query_lower):
            if len(word) <= 2 or word in AGGREGATION_STOP_WORDS or word in entities:
                continue
            if len(word) >= 4:
                entities.append(word)

        seen: set[str] = set()
        unique_entities: List[str] = []
        for entity in entities:
            if entity in seen:
                continue
            seen.add(entity)
            unique_entities.append(entity)
        return unique_entities

    def _extract_query_keywords(self, query_text: str) -> List[str]:
        keywords: List[str] = []
        for word in query_text.lower().split():
            clean_word = word.strip("?.,!'\"()[]{}:;")
            if len(clean_word) > 2 and clean_word not in QUESTION_STOP_WORDS:
                keywords.append(clean_word)
        return keywords

    def _detect_time_anchor(self, query_text: str) -> TimeAnchor:
        query_lower = query_text.lower()

        if _contains_any(query_lower, RECENT_ANCHOR_KEYWORDS):
            return TimeAnchor.RECENT
        if _contains_any(query_lower, EARLIEST_ANCHOR_KEYWORDS):
            return TimeAnchor.EARLIEST
        if _contains_any(query_lower, SPAN_ANCHOR_KEYWORDS):
            return TimeAnchor.SPAN
        return TimeAnchor.NONE
