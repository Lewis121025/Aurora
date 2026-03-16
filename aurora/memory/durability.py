"""持久度信号抽取模块。

从用户输入和认知结果推断片段的持久度（durability）。
高持久度片段在沉积清理中受保护，在检索中获得加权。

信号类型：
- preference: 用户长期偏好（"我喜欢…"、"我不喜欢…"）
- fact: 用户自我暴露的重要事实（"我是…"、"我有…"）
- commitment: Aurora 或用户做出的承诺（"我会…"、"我答应…"）
- boundary: 已建立的边界（"不要…"、"请不要…"）
"""
from __future__ import annotations

import re

_PREFERENCE_PATTERNS = re.compile(
    r"(?:我(?:喜欢|不喜欢|偏好|讨厌|爱|不爱|更想|希望|不想))"
    r"|(?:i (?:like|dislike|prefer|love|hate|don'?t like|don'?t want))",
    re.IGNORECASE,
)

_FACT_PATTERNS = re.compile(
    r"(?:我(?:是|有|在|住在|来自|叫|正在|曾经|毕业于|工作在))"
    r"|(?:i (?:am|have|live|work|study|was born|come from|my name))",
    re.IGNORECASE,
)

_COMMITMENT_PATTERNS = re.compile(
    r"(?:我(?:会|答应|保证|承诺|一定|不会再))"
    r"|(?:i (?:will|promise|won'?t|shall|commit))",
    re.IGNORECASE,
)

_BOUNDARY_PATTERNS = re.compile(
    r"(?:不要|请不要|别再|停止|边界|不可以|不允许)"
    r"|(?:don'?t|stop|boundary|never|do not)",
    re.IGNORECASE,
)

_PATTERN_WEIGHTS: tuple[tuple[re.Pattern[str], float], ...] = (
    (_BOUNDARY_PATTERNS, 0.72),
    (_COMMITMENT_PATTERNS, 0.68),
    (_FACT_PATTERNS, 0.56),
    (_PREFERENCE_PATTERNS, 0.52),
)


def estimate_durability(
    user_text: str,
    aurora_move: str,
    is_boundary_event: bool,
) -> float:
    """从用户输入和认知结果估算持久度。

    Args:
        user_text: 用户原始输入。
        aurora_move: Aurora 行为选择。
        is_boundary_event: 是否为边界事件。

    Returns:
        持久度值（0.0–1.0）。
    """
    score = 0.0
    for pattern, weight in _PATTERN_WEIGHTS:
        if pattern.search(user_text):
            score = max(score, weight)

    if is_boundary_event or aurora_move == "boundary":
        score = max(score, 0.72)
    if aurora_move == "repair":
        score = max(score, 0.48)

    return min(score, 1.0)
