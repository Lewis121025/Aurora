"""张力队列模块。

实现张力队列（TensionQueue），管理未解决的对话悬案：
- Priority Queue with 半衰期
- 紧迫度自动衰减
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(order=True, slots=True)
class TensionItem:
    """张力条目。

    Attributes:
        topic: 话题。
        urgency: 紧迫度（0.0-1.0）。
        halflife_hours: 半衰期（小时）。
        prompt: 主动抛出的提示语。
        created_at: 创建时间戳。
    """

    topic: str
    urgency: float = field(compare=False)
    halflife_hours: float = field(compare=False)
    prompt: str = field(compare=False)
    created_at: float = field(compare=False)

    def current_urgency(self, now_ts: float) -> float:
        """计算当前紧迫度（含衰减）。

        Args:
            now_ts: 当前时间戳。

        Returns:
            当前紧迫度。
        """
        hours_elapsed = (now_ts - self.created_at) / 3600
        decay = 0.5 ** (hours_elapsed / self.halflife_hours)
        return self.urgency * decay


class TensionQueue:
    """张力队列。

    管理未解决的对话悬案，支持：
    - push: 压入新悬案
    - pop: 弹出最紧迫的悬案
    - decay: 紧迫度衰减

    Attributes:
        _heap: 最小堆（按 -urgency 排序，最紧迫的在前）。
    """

    def __init__(self) -> None:
        self._heap: list[tuple[float, TensionItem]] = []

    def push(
        self,
        topic: str,
        urgency: float,
        halflife_hours: float,
        prompt: str,
        now_ts: float,
    ) -> None:
        """压入新悬案。

        Args:
            topic: 话题。
            urgency: 紧迫度（0.0-1.0）。
            halflife_hours: 半衰期（小时）。
            prompt: 提示语。
            now_ts: 当前时间戳。
        """
        item = TensionItem(
            topic=topic,
            urgency=urgency,
            halflife_hours=halflife_hours,
            prompt=prompt,
            created_at=now_ts,
        )
        heapq.heappush(self._heap, (-urgency, item))

    def pop(self, now_ts: float) -> TensionItem | None:
        """弹出最紧迫的悬案。

        Args:
            now_ts: 当前时间戳。

        Returns:
            最紧迫的张力条目或 None。
        """
        while self._heap:
            _, item = heapq.heappop(self._heap)
            if item.current_urgency(now_ts) > 0.1:
                return item
        return None

    def peek(self, now_ts: float) -> TensionItem | None:
        """查看最紧迫的悬案（不移除）。

        Args:
            now_ts: 当前时间戳。

        Returns:
            最紧迫的张力条目或 None。
        """
        valid_items = [
            (-neg_urg, item) for neg_urg, item in self._heap if item.current_urgency(now_ts) > 0.1
        ]
        if not valid_items:
            return None
        _, item = min(valid_items, key=lambda x: x[0])
        return item

    def to_prompt_segment(self, now_ts: float) -> str:
        """转换为 System Prompt 片段。

        Args:
            now_ts: 当前时间戳。

        Returns:
            格式化的张力队列字符串。
        """
        active = [item for _, item in self._heap if item.current_urgency(now_ts) > 0.1]
        if not active:
            return "[TENSION_QUEUE]\n  （无悬案）"

        lines = ["[TENSION_QUEUE]"]
        for item in active[:3]:
            urgency = item.current_urgency(now_ts)
            lines.append(f'  - topic: "{item.topic}"')
            lines.append(f"    urgency: {urgency:.2f}")
            lines.append(f'    prompt: "{item.prompt}"')
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._heap)

    def __iter__(self) -> Iterable[TensionItem]:
        return (item for _, item in self._heap)
