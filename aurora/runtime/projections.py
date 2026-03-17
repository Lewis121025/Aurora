"""Aurora v2 运行时投影与激活规则。"""

from __future__ import annotations

import re

from aurora.relation.state import to_prompt_segment as relation_prompt
from aurora.relation.tension import top_open_loops, to_prompt_segment as loop_prompt
from aurora.runtime.contracts import OpenLoop, RecallHit, RelationField

_RECALL_PATTERNS = (
    "记得",
    "之前",
    "上次",
    "以前",
    "我说过",
    "叫什么",
    "住在哪",
    "住在哪里",
    "哪天",
    "什么时候",
    "which",
    "where",
    "when",
    "what did i",
    "remember",
    "recall",
    "earlier",
    "previously",
)

_LOOP_PATTERNS = ("继续", "接着", "那个", "还没", "承诺", "提醒", "resolve", "finish")
_FACT_PATTERNS = re.compile(r"\b(name|variable|address|called|live|lives|born|date)\b")


def choose_archive_activation(text: str, loops: tuple[OpenLoop, ...]) -> bool:
    """用硬规则决定本轮是否需要冷记忆。"""
    lowered = text.lower()
    if any(pattern in lowered for pattern in _RECALL_PATTERNS):
        return True
    if _FACT_PATTERNS.search(lowered):
        return True
    if loops and any(pattern in text for pattern in _LOOP_PATTERNS):
        return True
    return False


def build_memory_projection(
    field: RelationField,
    loops: tuple[OpenLoop, ...],
    recall_hits: tuple[RecallHit, ...],
    recent_turns: tuple[str, ...],
    now_ts: float,
) -> tuple[str, str, tuple[RecallHit, ...], tuple[str, ...]]:
    """构建热路径上下文投影。"""
    active_loops = top_open_loops(loops, now_ts)
    return (
        relation_prompt(field),
        loop_prompt(active_loops, now_ts),
        recall_hits,
        recent_turns[-4:],
    )
