"""
AURORA 时间范围过滤器
========================

用于时间推理优化的时间感知查询扩展。

第一性原理：
- 时间查询受益于时间范围预过滤
- 对时间查询的搜索空间减少 40-60%
- 改进时间推理准确性 7-11%（论文发现）

核心思想：从查询推断时间范围，在语义搜索前过滤候选项。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable
import re

from aurora.core.constants import (
    EARLIEST_ANCHOR_KEYWORDS,
    RECENT_ANCHOR_KEYWORDS,
    SPAN_ANCHOR_KEYWORDS,
)


@dataclass
class TimeRange:
    """用于过滤候选项的时间范围规范。

    属性：
        start: 开始时间戳（包含）。None 表示无下界。
        end: 结束时间戳（包含）。None 表示无上界。
        anchor_event: 可选的事件文本以锚定范围
        relation: 时间关系类型（"first"、"last"、"before"、"after"、"during"、"any"）
    """
    start: Optional[float] = None  # timestamp
    end: Optional[float] = None
    anchor_event: Optional[str] = None
    relation: str = "any"  # "before", "after", "during", "first", "last", "any"


class TimeRangeExtractor:
    """从查询文本中提取时间范围以进行预过滤。

    该类通过从自然语言查询推断时间约束来实现时间感知查询扩展。
    提取的时间范围用于在昂贵的语义搜索操作之前过滤候选项。

    示例：
        >>> extractor = TimeRangeExtractor()
        >>> time_range = extractor.extract("最早学了什么？", events_timeline)
        >>> # 返回 TimeRange，relation="first"，end=earliest_timestamp+86400
    """

    # 时间锚点模式（匹配 field_retriever.py 中的常量）
    ANCHOR_PATTERNS = {
        "first": list(EARLIEST_ANCHOR_KEYWORDS),
        "last": list(RECENT_ANCHOR_KEYWORDS),
        "span": list(SPAN_ANCHOR_KEYWORDS),
    }

    # 相对时间模式（天数）
    RELATIVE_PATTERNS = {
        "last_week": r"(last week|上周|上星期)",
        "last_month": r"(last month|上个月|上月)",
        "yesterday": r"(yesterday|昨天)",
        "today": r"(today|今天)",
    }
    
    def extract(
        self,
        query: str,
        events_timeline: Optional[List[Tuple[str, float]]] = None
    ) -> TimeRange:
        """从查询文本中提取时间范围。

        参数：
            query: 要分析的查询文本
            events_timeline: 可选的 (event_text, timestamp) 元组列表
                用于锚点事件解析。如果为 None，使用简单的模式匹配。

        返回：
            TimeRange 对象，包含推断的时间约束
        """
        query_lower = query.lower()
        events_timeline = events_timeline or []

        # 从锚点模式检测时间关系
        for relation, keywords in self.ANCHOR_PATTERNS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return self._resolve_anchor(query, relation, events_timeline)

        # 检测相对时间模式
        for pattern_name, pattern in self.RELATIVE_PATTERNS.items():
            if re.search(pattern, query_lower, re.IGNORECASE):
                return self._resolve_relative_time(pattern_name, events_timeline)

        # 默认：无时间约束
        return TimeRange(relation="any")
    
    def _resolve_anchor(
        self,
        query: str,
        relation: str,
        events_timeline: List[Tuple[str, float]]
    ) -> TimeRange:
        """将时间锚点解析为特定的时间范围。

        参数：
            query: 原始查询文本
            relation: 检测到的关系（"first"、"last"、"span"）
            events_timeline: (event_text, timestamp) 元组列表

        返回：
            TimeRange，包含已解析的开始/结束时间戳
        """
        if not events_timeline:
            # 没有可用的时间线，返回仅包含关系的范围
            return TimeRange(relation=relation)

        timestamps = [ts for _, ts in events_timeline]

        if relation == "first":
            # 返回最早的时间段（前 24 小时）
            earliest = min(timestamps)
            return TimeRange(
                end=earliest + 86400,  # +1 天缓冲
                relation="first"
            )

        elif relation == "last":
            # 返回最新的时间段（最后 24 小时）
            latest = max(timestamps)
            return TimeRange(
                start=latest - 86400,  # -1 天缓冲
                relation="last"
            )

        elif relation == "span":
            # 跨度查询需要完整范围（无过滤）
            return TimeRange(relation="span")

        return TimeRange(relation=relation)
    
    def _resolve_relative_time(
        self,
        pattern_name: str,
        events_timeline: List[Tuple[str, float]]
    ) -> TimeRange:
        """解析相对时间模式（昨天、上周等）。

        参数：
            pattern_name: 检测到的模式名称
            events_timeline: (event_text, timestamp) 元组列表

        返回：
            TimeRange，包含已解析的相对时间边界
        """
        if not events_timeline:
            return TimeRange(relation="any")

        # 从最新事件获取当前时间
        latest_ts = max(ts for _, ts in events_timeline)

        if pattern_name == "yesterday":
            # 昨天 = 24-48 小时前
            start = latest_ts - 2 * 86400
            end = latest_ts - 86400
            return TimeRange(start=start, end=end, relation="during")

        elif pattern_name == "today":
            # 今天 = 最后 24 小时
            return TimeRange(start=latest_ts - 86400, relation="during")

        elif pattern_name == "last_week":
            # 上周 = 7-14 天前
            start = latest_ts - 14 * 86400
            end = latest_ts - 7 * 86400
            return TimeRange(start=start, end=end, relation="during")

        elif pattern_name == "last_month":
            # 上月 = 30-60 天前
            start = latest_ts - 60 * 86400
            end = latest_ts - 30 * 86400
            return TimeRange(start=start, end=end, relation="during")

        return TimeRange(relation="any")
    
    def filter_by_range(
        self,
        candidates: List[Tuple[str, float, float]],  # [(id, score, timestamp), ...]
        time_range: TimeRange,
        get_timestamp: Optional[Callable[[str], float]] = None  # 获取节点 ID 时间戳的函数
    ) -> List[Tuple[str, float, float]]:
        """按时间范围过滤候选项。

        参数：
            candidates: (id, score, timestamp) 元组列表
            time_range: 要过滤的 TimeRange
            get_timestamp: 获取节点 ID 时间戳的函数（如果元组中没有则回退）

        返回：
            在时间范围内的过滤候选项列表
        """
        if time_range.relation == "any" or time_range.relation == "span":
            # "any" 或 "span" 关系不进行过滤
            return candidates

        filtered = []
        for item in candidates:
            if len(item) == 3:
                cid, score, ts = item
            else:
                # 回退：从函数获取时间戳
                cid, score = item[0], item[1]
                ts = get_timestamp(cid) if get_timestamp else 0.0

            # 应用时间边界
            if time_range.start is not None and ts < time_range.start:
                continue
            if time_range.end is not None and ts > time_range.end:
                continue

            filtered.append((cid, score, ts))

        # 按关系类型排序
        if time_range.relation == "first":
            # 升序排列（最早优先）
            filtered.sort(key=lambda x: x[2])
        elif time_range.relation == "last":
            # 降序排列（最新优先）
            filtered.sort(key=lambda x: -x[2])

        return filtered
