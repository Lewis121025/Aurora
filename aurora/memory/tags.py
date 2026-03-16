"""记忆标签提取模块。

从片段表面文本中提取关键词标签，支持：
- 拉丁字母单词（3 字母以上）
- CJK 字符（1-4 字）
按出现频率排序，最多保留 12 个标签。
"""
from __future__ import annotations

import re
from collections import Counter


def extract_tags(text: str) -> tuple[str, ...]:
    """从文本提取标签。

    提取规则：
    - 拉丁字母单词：3 字母以上，转小写
    - CJK 字符：1-4 字

    Args:
        text: 表面文本。

    Returns:
        标签元组，按频率降序排列，最多 12 个。
    """
    # 提取拉丁字母单词
    latin = re.findall(r"[A-Za-z]{3,}", text.lower())
    # 提取 CJK 字符
    cjk = re.findall(r"[\u4e00-\u9fff]{1,4}", text)

    # 合并并去空
    merged = [token.strip() for token in latin + cjk if token.strip()]

    # 无有效标签时使用默认值
    if not merged:
        merged = [text[:12].strip() or "moment"]

    # 按频率排序，返回前 12 个
    counter = Counter(merged)
    return tuple(sorted(token for token, _ in counter.most_common(12)))
